from collections import defaultdict

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
torch.manual_seed(0)

from dataset import construct_dataset
from network import *

from .base_model import BaseModel
from .geometry import Pose, ViewRendering
from .losses import MultiCamLoss, SingleCamLoss

from .gaussian import GaussianNetwork, depth2pc, pts2render, focal2fov, getProjectionMatrix, getWorld2View2, rotate_sh
from einops import rearrange

_NO_DEVICE_KEYS = ['idx', 'dataset_idx', 'sensor_name', 'filename', 'token']


class DrivingForwardModel(BaseModel):
    def __init__(self, cfg, rank):
        super(DrivingForwardModel, self).__init__(cfg)
        self.rank = rank
        self.read_config(cfg)
        self.prepare_dataset(cfg, rank)
        self.models = self.prepare_model(cfg, rank)   
        self.losses = self.init_losses(cfg, rank)        
        self.view_rendering, self.pose = self.init_geometry(cfg, rank) 
        self.set_optimizer()
        
        if self.pretrain and rank == 0:
            self.load_weights()

        self.left_cam_dict = {2:0, 0:1, 4:2, 1:3, 5:4, 3:5}
        self.right_cam_dict = {0:2, 1:0, 2:4, 3:1, 4:5, 5:3}
        
    def read_config(self, cfg):    
        for attr in cfg.keys(): 
            for k, v in cfg[attr].items():
                setattr(self, k, v)
                
    def init_geometry(self, cfg, rank):
        view_rendering = ViewRendering(cfg, rank)
        pose = Pose(cfg)
        return view_rendering, pose
        
    def init_losses(self, cfg, rank):
        if self.spatio_temporal or self.spatio:
            loss_model = MultiCamLoss(cfg, rank)
        else :
            loss_model = SingleCamLoss(cfg, rank)
        return loss_model
        
    def prepare_model(self, cfg, rank):
        models = {}
        models['pose_net'] = self.set_posenet(cfg)        
        models['depth_net'] = self.set_depthnet(cfg)
        if self.gaussian:
            models['gs_net'] = self.set_gaussiannet(cfg)

        return models

    def set_posenet(self, cfg):
        return PoseNetwork(cfg).cuda()
        
    def set_depthnet(self, cfg):
        return DepthNetwork(cfg).cuda()

    def set_gaussiannet(self, cfg):
        legacy_114x228 = (int(self.height), int(self.width)) == (114, 228)
        return GaussianNetwork(
            rgb_dim=3,
            depth_dim=1,
            legacy_114x228=legacy_114x228).cuda()

    def prepare_dataset(self, cfg, rank):
        if rank == 0:
            print('### Preparing Datasets')
        
        if self.mode == 'train':
            self.set_train_dataloader(cfg, rank)
            if rank == 0 :
                self.set_val_dataloader(cfg)
                
        if self.mode == 'eval':
            self.set_eval_dataloader(cfg)

    def set_train_dataloader(self, cfg, rank):                 
        # jittering augmentation and image resizing for the training data
        _augmentation = {
            'image_shape': (int(self.height), int(self.width)), 
            'jittering': (0.2, 0.2, 0.2, 0.05),
            'crop_train_borders': (),
            'crop_eval_borders': ()
        }

        # construct train dataset
        train_dataset = construct_dataset(cfg, 'train', **_augmentation)

        dataloader_opts = {
            'batch_size': self.batch_size,
            'shuffle': True,
            'num_workers': self.num_workers,
            'pin_memory': True,
            'drop_last': True
        }

        self._dataloaders['train'] = DataLoader(train_dataset, **dataloader_opts)
        num_train_samples = len(train_dataset)    
        self.num_total_steps = num_train_samples // (self.batch_size * self.world_size) * self.num_epochs

    def set_val_dataloader(self, cfg):         
        # Image resizing for the validation data
        _augmentation = {
            'image_shape': (int(self.height), int(self.width)),
            'jittering': (0.0, 0.0, 0.0, 0.0),
            'crop_train_borders': (),
            'crop_eval_borders': ()
        }

        # construct validation dataset
        val_dataset = construct_dataset(cfg, 'val', **_augmentation)

        dataloader_opts = {
            'batch_size': self.batch_size,
            'shuffle': False,
            'num_workers': 0,
            'pin_memory': True,
            'drop_last': True
        }

        self._dataloaders['val']  = DataLoader(val_dataset, **dataloader_opts)
    
    def set_eval_dataloader(self, cfg):  
        # Image resizing for the validation data
        _augmentation = {
            'image_shape': (int(self.height), int(self.width)),
            'jittering': (0.0, 0.0, 0.0, 0.0),
            'crop_train_borders': (),
            'crop_eval_borders': ()
        }

        dataloader_opts = {
            'batch_size': self.eval_batch_size,
            'shuffle': False,
            'num_workers': self.eval_num_workers,
            'pin_memory': True,
            'drop_last': True
        }

        eval_dataset = construct_dataset(cfg, 'eval', **_augmentation)

        self._dataloaders['eval'] = DataLoader(eval_dataset, **dataloader_opts)

    def set_optimizer(self):
        parameters_to_train = []
        for v in self.models.values():
            parameters_to_train += list(v.parameters())

        self.optimizer = optim.Adam(
        parameters_to_train, 
            self.learning_rate
        )

        self.lr_scheduler = optim.lr_scheduler.StepLR(
            self.optimizer, 
            self.scheduler_step_size,
            0.1
        )
    
    def process_batch(self, inputs, rank):
        """
        Pass a minibatch through the network and generate images, depth maps, and losses.
        """
        for key, ipt in inputs.items():
            if key not in _NO_DEVICE_KEYS:
                if 'context' in key:
                    inputs[key] = [ipt[k].float().to(rank) for k in range(len(inputs[key]))]
                if 'ego_pose' in key:
                    inputs[key] = [ipt[k].float().to(rank) for k in range(len(inputs[key]))]
                else:
                    inputs[key] = ipt.float().to(rank)   

        outputs = self.estimate(inputs)
        losses = self.compute_losses(inputs, outputs)
        return outputs, losses  

    def estimate(self, inputs):
        """
        This function estimates the outputs of the network.
        """          
        # pre-calculate inverse of the extrinsic matrix
        inputs['extrinsics_inv'] = torch.inverse(inputs['extrinsics'])
        
        # init dictionary 
        outputs = {}
        for cam in range(self.num_cams):
            outputs[('cam', cam)] = {}

        pose_pred = self.predict_pose(inputs)
        depth_feats = self.predict_depth(inputs)

        for cam in range(self.num_cams):       
            if self.mode != 'train':
                outputs[('cam', cam)].update({('cam_T_cam', 0, 1): inputs[('cam_T_cam', 0, 1)][:, cam, ...]})
                outputs[('cam', cam)].update({('cam_T_cam', 0, -1): inputs[('cam_T_cam', 0, -1)][:, cam, ...]}) 
            elif self.mode == 'train':
                outputs[('cam', cam)].update(pose_pred[('cam', cam)])                
            outputs[('cam', cam)].update(depth_feats[('cam', cam)])
            
        self.compute_depth_maps(inputs, outputs)
        return outputs

    def predict_pose(self, inputs):      
        """
        This function predicts poses.
        """          
        net = self.models['pose_net']
        
        pose = self.pose.compute_pose(net, inputs)
        return pose

    def predict_depth(self, inputs):
        """
        This function predicts disparity maps.
        """                  
        net = self.models['depth_net']

        depth_feats = net(inputs)
        return depth_feats
    
    def compute_depth_maps(self, inputs, outputs):     
        """
        This function computes depth map for each viewpoint.
        """                  
        source_scale = 0
        for cam in range(self.num_cams):
            ref_K = inputs[('K', source_scale)][:, cam, ...]
            for scale in self.scales:
                disp = outputs[('cam', cam)][('disp', scale)]
                outputs[('cam', cam)][('depth', 0, scale)] = self.to_depth(disp, ref_K)
                if self.novel_view_mode == 'MF':
                    disp_last = outputs[('cam', cam)][('disp', -1, scale)]
                    outputs[('cam', cam)][('depth', -1, scale)] = self.to_depth(disp_last, ref_K)
                    disp_next = outputs[('cam', cam)][('disp', 1, scale)]
                    outputs[('cam', cam)][('depth', 1, scale)] = self.to_depth(disp_next, ref_K)
    
    def to_depth(self, disp_in, K_in):        
        """
        This function transforms disparity value into depth map while multiplying the value with the focal length.
        """
        min_disp = 1/self.max_depth
        max_disp = 1/self.min_depth
        disp_range = max_disp-min_disp

        disp_in = F.interpolate(disp_in, [self.height, self.width], mode='bilinear', align_corners=False)
        disp = min_disp + disp_range * disp_in
        depth = 1/disp
        return depth * K_in[:, 0:1, 0:1].unsqueeze(2)/self.focal_length_scale

    def get_gaussian_data(self, inputs, outputs, cam):
        """
        This function computes gaussian data for each viewpoint.
        """
        bs, _, height, width = inputs[('color', 0, 0)][:, cam, ...].shape
        zfar = self.max_depth
        znear = 0.01

        if self.novel_view_mode == 'MF':
            for frame_id in self.frame_ids:
                if frame_id == 0:
                    outputs[('cam', cam)][('e2c_extr', frame_id, 0)] = inputs['extrinsics_inv'][:, cam, ...]
                    outputs[('cam', cam)][('c2e_extr', frame_id, 0)] = inputs['extrinsics'][:, cam, ...]
                    FovX_list = []
                    FovY_list = []
                    world_view_transform_list = []
                    full_proj_transform_list = []
                    camera_center_list = []
                    for i in range(bs):
                        intr = inputs[('K', 0)][:, cam, ...][i,:]
                        extr = inputs['extrinsics_inv'][:, cam, ...][i,:]
                        FovX = focal2fov(intr[0, 0], width)
                        FovY = focal2fov(intr[1, 1], height)
                        projection_matrix = getProjectionMatrix(znear=znear, zfar=zfar, K=intr, h=height, w=width).transpose(0, 1).cuda()
                        world_view_transform = torch.tensor(extr).transpose(0, 1).cuda()
                        # full_proj_transform: (E^T K^T) = (K E)^T
                        full_proj_transform = (world_view_transform.unsqueeze(0).bmm(projection_matrix.unsqueeze(0))).squeeze(0)
                        camera_center = world_view_transform.inverse()[3, :3] 

                        FovX_list.append(FovX)
                        FovY_list.append(FovY)
                        world_view_transform_list.append(world_view_transform.unsqueeze(0))
                        full_proj_transform_list.append(full_proj_transform.unsqueeze(0))
                        camera_center_list.append(camera_center.unsqueeze(0))

                    outputs[('cam', cam)][('FovX', frame_id, 0)] = torch.tensor(FovX_list).cuda()
                    outputs[('cam', cam)][('FovY', frame_id, 0)] = torch.tensor(FovY_list).cuda()
                    outputs[('cam', cam)][('world_view_transform', frame_id, 0)] = torch.cat(world_view_transform_list, dim=0)
                    outputs[('cam', cam)][('full_proj_transform', frame_id, 0)] = torch.cat(full_proj_transform_list, dim=0)
                    outputs[('cam', cam)][('camera_center', frame_id, 0)] = torch.cat(camera_center_list, dim=0)
                else:
                    outputs[('cam', cam)][('e2c_extr', frame_id, 0)] = \
                        torch.matmul(outputs[('cam', cam)][('cam_T_cam', 0, frame_id)], inputs['extrinsics_inv'][:, cam, ...])
                    outputs[('cam', cam)][('c2e_extr', frame_id, 0)] = \
                        torch.matmul(inputs['extrinsics'][:, cam, ...], torch.inverse(outputs[('cam', cam)][('cam_T_cam', 0, frame_id)]))
                outputs[('cam', cam)][('xyz', frame_id, 0)] = depth2pc(outputs[('cam', cam)][('depth', frame_id, 0)], outputs[('cam', cam)][('e2c_extr', frame_id, 0)], inputs[('K', 0)][:, cam, ...])
                valid = outputs[('cam', cam)][('depth', frame_id, 0)] != 0.0
                outputs[('cam', cam)][('pts_valid', frame_id, 0)] = valid.view(bs, -1)
                rot_maps, scale_maps, opacity_maps, sh_maps = \
                    self.gs_net(inputs[('color', frame_id, 0)][:, cam, ...], outputs[('cam', cam)][('depth', frame_id, 0)], outputs[('cam', cam)][('img_feat', frame_id, 0)])
                c2w_rotations = rearrange(outputs[('cam', cam)][('c2e_extr', frame_id, 0)][..., :3, :3], "k i j -> k () () () i j")
                sh_maps = rotate_sh(sh_maps, c2w_rotations[..., None, :, :])
                outputs[('cam', cam)][('rot_maps', frame_id, 0)] = rot_maps
                outputs[('cam', cam)][('scale_maps', frame_id, 0)] = scale_maps
                outputs[('cam', cam)][('opacity_maps', frame_id, 0)] = opacity_maps
                outputs[('cam', cam)][('sh_maps', frame_id, 0)] = sh_maps
        elif self.novel_view_mode == 'SF':
            frame_id = 0
            outputs[('cam', cam)][('e2c_extr', frame_id, 0)] = inputs['extrinsics_inv'][:, cam, ...]
            outputs[('cam', cam)][('c2e_extr', frame_id, 0)] = inputs['extrinsics'][:, cam, ...]
            outputs[('cam', cam)][('xyz', frame_id, 0)] = depth2pc(outputs[('cam', cam)][('depth', frame_id, 0)], outputs[('cam', cam)][('e2c_extr', frame_id, 0)], inputs[('K', 0)][:, cam, ...])
            valid = outputs[('cam', cam)][('depth', frame_id, 0)] != 0.0
            outputs[('cam', cam)][('pts_valid', frame_id, 0)] = valid.view(bs, -1)
            rot_maps, scale_maps, opacity_maps, sh_maps = \
                self.gs_net(inputs[('color', frame_id, 0)][:, cam, ...], outputs[('cam', cam)][('depth', frame_id, 0)], outputs[('cam', cam)][('img_feat', frame_id, 0)])
            c2w_rotations = rearrange(outputs[('cam', cam)][('c2e_extr', frame_id, 0)][..., :3, :3], "k i j -> k () () () i j")
            sh_maps = rotate_sh(sh_maps, c2w_rotations[..., None, :, :])
            outputs[('cam', cam)][('rot_maps', frame_id, 0)] = rot_maps
            outputs[('cam', cam)][('scale_maps', frame_id, 0)] = scale_maps
            outputs[('cam', cam)][('opacity_maps', frame_id, 0)] = opacity_maps
            outputs[('cam', cam)][('sh_maps', frame_id, 0)] = sh_maps

            # novel view
            for frame_id in self.frame_ids[1:]:
                outputs[('cam', cam)][('e2c_extr', frame_id, 0)] = \
                    torch.matmul(outputs[('cam', cam)][('cam_T_cam', 0, frame_id)], inputs['extrinsics_inv'][:, cam, ...])
                outputs[('cam', cam)][('c2e_extr', frame_id, 0)] = \
                    torch.matmul(inputs['extrinsics'][:, cam, ...], torch.inverse(outputs[('cam', cam)][('cam_T_cam', 0, frame_id)]))
                
                FovX_list = []
                FovY_list = []
                world_view_transform_list = []
                full_proj_transform_list = []
                camera_center_list = []
                for i in range(bs):
                    intr = inputs[('K', 0)][:, cam, ...][i,:]
                    extr = inputs['extrinsics_inv'][:, cam, ...][i,:]
                    T_i = outputs[('cam', cam)][('cam_T_cam', 0, frame_id)][i,:]
                    FovX = focal2fov(intr[0, 0], width)
                    FovY = focal2fov(intr[1, 1], height)
                    projection_matrix = getProjectionMatrix(znear=znear, zfar=zfar, K=intr, h=height, w=width).transpose(0, 1).cuda()
                    world_view_transform = torch.matmul(T_i, torch.tensor(extr).cuda()).transpose(0, 1)
                    # full_proj_transform: (E^T K^T) = (K E)^T
                    full_proj_transform = (world_view_transform.unsqueeze(0).bmm(projection_matrix.unsqueeze(0))).squeeze(0)
                    camera_center = world_view_transform.inverse()[3, :3] 
                    FovX_list.append(FovX)
                    FovY_list.append(FovY)
                    world_view_transform_list.append(world_view_transform.unsqueeze(0))
                    full_proj_transform_list.append(full_proj_transform.unsqueeze(0))
                    camera_center_list.append(camera_center.unsqueeze(0))
                outputs[('cam', cam)][('FovX', frame_id, 0)] = torch.tensor(FovX_list).cuda()
                outputs[('cam', cam)][('FovY', frame_id, 0)] = torch.tensor(FovY_list).cuda()
                outputs[('cam', cam)][('world_view_transform', frame_id, 0)] = torch.cat(world_view_transform_list, dim=0)
                outputs[('cam', cam)][('full_proj_transform', frame_id, 0)] = torch.cat(full_proj_transform_list, dim=0)
                outputs[('cam', cam)][('camera_center', frame_id, 0)] = torch.cat(camera_center_list, dim=0)
    
    def compute_losses(self, inputs, outputs):
        """
        This function computes losses.
        """          
        losses = 0
        loss_fn = defaultdict(list)
        loss_mean = defaultdict(float)

        # compute gaussian data
        if self.gaussian:
            self.gs_net = self.models['gs_net']
            for cam in range(self.num_cams):
                self.get_gaussian_data(inputs, outputs, cam)

        # generate image and compute loss per cameara
        for cam in range(self.num_cams):
            self.pred_cam_imgs(inputs, outputs, cam)
            if self.gaussian:
                self.pred_gaussian_imgs(inputs, outputs, cam)
            cam_loss, loss_dict = self.losses(inputs, outputs, cam)
            
            losses += cam_loss  
            for k, v in loss_dict.items():
                loss_fn[k].append(v)

        losses /= self.num_cams
        
        for k in loss_fn.keys():
            loss_mean[k] = sum(loss_fn[k]) / float(len(loss_fn[k]))

        loss_mean['total_loss'] = losses        
        return loss_mean

    def pred_cam_imgs(self, inputs, outputs, cam):
        """
        This function renders projected images using camera parameters and depth information.
        """                  
        rel_pose_dict = self.pose.compute_relative_cam_poses(inputs, outputs, cam)
        self.view_rendering(inputs, outputs, cam, rel_pose_dict) 

    def pred_gaussian_imgs(self, inputs, outputs, cam):
        if self.novel_view_mode == 'MF':
            outputs[('cam', cam)][('gaussian_color', 0, 0)] = \
                pts2render(inputs=inputs, 
                           outputs=outputs,
                           cam_num=self.num_cams, 
                           novel_cam=cam,
                           novel_frame_id=0,
                           bg_color=[1.0, 1.0, 1.0],
                           mode=self.novel_view_mode)
        elif self.novel_view_mode == 'SF':
            for novel_frame_id in self.frame_ids[1:]:
                outputs[('cam', cam)][('gaussian_color', novel_frame_id, 0)] = \
                    pts2render(inputs=inputs, 
                               outputs=outputs, 
                               cam_num=self.num_cams, 
                               novel_cam=cam,
                               novel_frame_id=novel_frame_id, 
                               bg_color=[1.0, 1.0, 1.0],
                               mode=self.novel_view_mode)
