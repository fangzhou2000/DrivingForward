from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import upsample, conv2d, pack_cam_feat, unpack_cam_feat
from .volumetric_fusionnet import VFNet

from external.layers import ResnetEncoder

class DepthNetwork(nn.Module):
    """
    Depth fusion module
    """    
    def __init__(self, cfg):
        super(DepthNetwork, self).__init__()
        self.read_config(cfg)
        
        # feature encoder        
        # resnet feat: 64(1/2), 64(1/4), 128(1/8), 256(1/16), 512(1/32)        
        self.encoder = ResnetEncoder(self.num_layers, self.weights_init, 1) # number of layers, pretrained, number of input images
        del self.encoder.encoder.fc # del fc in weights_init
        enc_feat_dim = sum(self.encoder.num_ch_enc[self.fusion_level:]) 
        self.conv1x1 = conv2d(enc_feat_dim, self.fusion_feat_in_dim, kernel_size=1, padding_mode = 'reflect') 

        # fusion net
        fusion_feat_out_dim = self.encoder.num_ch_enc[self.fusion_level] 
        self.fusion_net = VFNet(cfg, self.fusion_feat_in_dim, fusion_feat_out_dim, model ='depth')
        
        # depth decoder
        num_ch_enc = self.encoder.num_ch_enc[:(self.fusion_level+1)] 
        num_ch_dec = [16, 32, 64, 128, 256]
        self.decoder = DepthDecoder(
            self.fusion_level,
            num_ch_enc,
            num_ch_dec,
            self.scales,
            use_skips=self.use_skips,
            output_size=(int(self.height), int(self.width)))
    
    def read_config(self, cfg):
        for attr in cfg.keys(): 
            for k, v in cfg[attr].items():
                setattr(self, k, v)

    def forward(self, inputs):
        '''
        dict_keys(['idx', 'sensor_name', 'filename', 'extrinsics', 'mask', 
        ('K', 0), ('inv_K', 0), ('color', 0, 0), ('color_aug', 0, 0), 
        ('K', 1), ('inv_K', 1), ('color', 0, 1), ('color_aug', 0, 1), 
        ('K', 2), ('inv_K', 2), ('color', 0, 2), ('color_aug', 0, 2), 
        ('K', 3), ('inv_K', 3), ('color', 0, 3), ('color_aug', 0, 3), 
        ('color', -1, 0), ('color_aug', -1, 0), ('color', 1, 0), ('color_aug', 1, 0), 'extrinsics_inv'])
        '''

        outputs = {}
        
        # dictionary initialize
        for cam in range(self.num_cams): # self.num_cames = 6
            outputs[('cam', cam)] = {} # outputs = {('cam', 0): {}, ..., ('cam', 5): {}}
        
        lev = self.fusion_level # 2
        
        # packed images for surrounding view
        sf_images = torch.stack([inputs[('color_aug', 0, 0)][:, cam, ...] for cam in range(self.num_cams)], 1) 
        if self.novel_view_mode == 'MF':
            sf_images_last = torch.stack([inputs[('color_aug', -1, 0)][:, cam, ...] for cam in range(self.num_cams)], 1)
            sf_images_next = torch.stack([inputs[('color_aug', 1, 0)][:, cam, ...] for cam in range(self.num_cams)], 1)
        packed_input = pack_cam_feat(sf_images) 
        if self.novel_view_mode == 'MF':
            packed_input_last = pack_cam_feat(sf_images_last)
            packed_input_next = pack_cam_feat(sf_images_next)
        
        # feature encoder
        packed_feats = self.encoder(packed_input) 
        if self.novel_view_mode == 'MF':
            packed_feats_last = self.encoder(packed_input_last)
            packed_feats_next = self.encoder(packed_input_next)
        # aggregate feature H / 2^(lev+1) x W / 2^(lev+1)
        _, _, up_h, up_w = packed_feats[lev].size() 
        
        packed_feats_list = packed_feats[lev:lev+1] \
                        + [F.interpolate(feat, [up_h, up_w], mode='bilinear', align_corners=True) for feat in packed_feats[lev+1:]]        
        if self.novel_view_mode == 'MF':
            packed_feats_last_list = packed_feats_last[lev:lev+1] \
                            + [F.interpolate(feat, [up_h, up_w], mode='bilinear', align_corners=True) for feat in packed_feats_last[lev+1:]]
            packed_feats_next_list = packed_feats_next[lev:lev+1] \
                            + [F.interpolate(feat, [up_h, up_w], mode='bilinear', align_corners=True) for feat in packed_feats_next[lev+1:]]                

        packed_feats_agg = self.conv1x1(torch.cat(packed_feats_list, dim=1)) 
        if self.novel_view_mode == 'MF':
            packed_feats_agg_last = self.conv1x1(torch.cat(packed_feats_last_list, dim=1))
            packed_feats_agg_next = self.conv1x1(torch.cat(packed_feats_next_list, dim=1))

        feats_agg = unpack_cam_feat(packed_feats_agg, self.batch_size, self.num_cams) 
        if self.novel_view_mode == 'MF':
            feats_agg_last = unpack_cam_feat(packed_feats_agg_last, self.batch_size, self.num_cams)
            feats_agg_next = unpack_cam_feat(packed_feats_agg_next, self.batch_size, self.num_cams)

        # fusion_net, backproject each feature into the 3D voxel space
        fusion_dict = self.fusion_net(inputs, feats_agg)
        if self.novel_view_mode == 'MF':
            fusion_dict_last = self.fusion_net(inputs, feats_agg_last)
            fusion_dict_next = self.fusion_net(inputs, feats_agg_next)

        feat_in = packed_feats[:lev] + [fusion_dict['proj_feat']]   
        img_feat = []
        for i in range(len(feat_in)):
            img_feat.append(unpack_cam_feat(feat_in[i], self.batch_size, self.num_cams))
        
        if self.novel_view_mode == 'MF':
            feat_in_last = packed_feats_last[:lev] + [fusion_dict_last['proj_feat']] 
            img_feat_last = []
            for i in range(len(feat_in_last)):
                img_feat_last.append(unpack_cam_feat(feat_in_last[i], self.batch_size, self.num_cams))
            
            feat_in_next = packed_feats_next[:lev] + [fusion_dict_next['proj_feat']] 
            img_feat_next = []
            for i in range(len(feat_in_next)):
                img_feat_next.append(unpack_cam_feat(feat_in_next[i], self.batch_size, self.num_cams))

        packed_depth_outputs = self.decoder(feat_in)  
        if self.novel_view_mode == 'MF':      
            packed_depth_outputs_last = self.decoder(feat_in_last)
            packed_depth_outputs_next = self.decoder(feat_in_next)

        depth_outputs = unpack_cam_feat(packed_depth_outputs, self.batch_size, self.num_cams) 
        if self.novel_view_mode == 'MF':
            depth_outputs_last = unpack_cam_feat(packed_depth_outputs_last, self.batch_size, self.num_cams)
            depth_outputs_next = unpack_cam_feat(packed_depth_outputs_next, self.batch_size, self.num_cams)

        for cam in range(self.num_cams):
            for k in depth_outputs.keys():
                outputs[('cam', cam)][k] = depth_outputs[k][:, cam, ...]
            outputs[('cam', cam)][('img_feat', 0, 0)] = [feat[:, cam, ...] for feat in img_feat] 
            if self.novel_view_mode == 'MF':
                outputs[('cam', cam)][('img_feat', -1, 0)] = [feat[:, cam, ...] for feat in img_feat_last] 
                outputs[('cam', cam)][('img_feat', 1, 0)] = [feat[:, cam, ...] for feat in img_feat_next]
                outputs[('cam', cam)][('disp', -1, 0)] = depth_outputs_last[('disp', 0)][:, cam, ...] 
                outputs[('cam', cam)][('disp', 1, 0)] = depth_outputs_next[('disp', 0)][:, cam, ...] 

        return outputs
    
        
class DepthDecoder(nn.Module):
    def __init__(self, level_in, num_ch_enc, num_ch_dec, scales=range(2),
                 use_skips=False, output_size=None):
        super(DepthDecoder, self).__init__()

        self.num_output_channels = 1
        self.scales = scales
        self.use_skips = use_skips
        self.output_size = output_size
        
        self.level_in = level_in
        self.num_ch_enc = num_ch_enc
        self.num_ch_dec = num_ch_dec

        self.convs = OrderedDict()
        for i in range(self.level_in, -1, -1):
            num_ch_in = self.num_ch_enc[-1] if i == self.level_in else self.num_ch_dec[i + 1]
            num_ch_out = self.num_ch_dec[i]
            self.convs[('upconv', i, 0)] = conv2d(num_ch_in, num_ch_out, kernel_size=3, nonlin = 'ELU')

            num_ch_in = self.num_ch_dec[i]
            if self.use_skips and i > 0:
                num_ch_in += self.num_ch_enc[i - 1]
            num_ch_out = self.num_ch_dec[i]
            self.convs[('upconv', i, 1)] = conv2d(num_ch_in, num_ch_out, kernel_size=3, nonlin = 'ELU')

        for s in self.scales:
            self.convs[('dispconv', s)] = conv2d(self.num_ch_dec[s], self.num_output_channels, 3, nonlin = None)

        self.decoder = nn.ModuleList(list(self.convs.values()))
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_features):
        outputs = {}
        
        # decode
        x = input_features[-1]
        for i in range(self.level_in, -1, -1):
            x = self.convs[('upconv', i, 0)](x)
            x = [upsample(x)]
            if self.use_skips and i > 0:
                x += [input_features[i - 1]]
            x = torch.cat(x, 1)
            x = self.convs[('upconv', i, 1)](x)
            if i in self.scales:
                disp = self.sigmoid(self.convs[('dispconv', i)](x))
                if self.output_size is not None:
                    disp = F.interpolate(
                        disp,
                        size=self.output_size,
                        mode='bilinear',
                        align_corners=True)
                outputs[('disp', i)] = disp
        return outputs
