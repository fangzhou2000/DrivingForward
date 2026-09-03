
import torch
from torch import nn
import torch.nn.functional as F
from .extractor import UnetExtractor, ResidualBlock
from einops import rearrange


class GaussianNetwork(nn.Module):
    def __init__(self, rgb_dim=3, depth_dim=1, norm_fn='group',
                 legacy_114x228=False):
        super().__init__()
        self.rgb_dims = [64, 64, 128]
        self.depth_dims = [32, 48, 96]
        self.decoder_dims = [48, 64, 96]
        self.head_dim = 32

        self.sh_degree = 4
        self.d_sh = (self.sh_degree + 1) ** 2

        self.register_buffer(
            "sh_mask",
            torch.ones((self.d_sh,), dtype=torch.float32),
            persistent=False,
        )
        for degree in range(1, self.sh_degree + 1):
            self.sh_mask[degree**2 : (degree + 1) ** 2] = 0.1 * 0.25**degree

        self.depth_encoder = UnetExtractor(
            in_channel=depth_dim,
            encoder_dim=self.depth_dims,
            legacy_114x228=legacy_114x228)

        self.decoder3 = nn.Sequential(
            ResidualBlock(self.rgb_dims[2]+self.depth_dims[2], self.decoder_dims[2], norm_fn=norm_fn),
            ResidualBlock(self.decoder_dims[2], self.decoder_dims[2], norm_fn=norm_fn)
        )

        self.decoder2 = nn.Sequential(
            ResidualBlock(self.rgb_dims[1]+self.depth_dims[1]+self.decoder_dims[2], self.decoder_dims[1], norm_fn=norm_fn),
            ResidualBlock(self.decoder_dims[1], self.decoder_dims[1], norm_fn=norm_fn)
        )

        self.decoder1 = nn.Sequential(
            ResidualBlock(self.rgb_dims[0]+self.depth_dims[0]+self.decoder_dims[1], self.decoder_dims[0], norm_fn=norm_fn),
            ResidualBlock(self.decoder_dims[0], self.decoder_dims[0], norm_fn=norm_fn)
        )
        self.out_conv = nn.Conv2d(self.decoder_dims[0]+rgb_dim+1, self.head_dim, kernel_size=3, padding=1)
        self.out_relu = nn.ReLU(inplace=True)

        self.rot_head = nn.Sequential(
            nn.Conv2d(self.head_dim, self.head_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.head_dim, 4, kernel_size=1),
        )
        self.scale_head = nn.Sequential(
            nn.Conv2d(self.head_dim, self.head_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.head_dim, 3, kernel_size=1),
            nn.Softplus(beta=100)
        )
        self.opacity_head = nn.Sequential(
            nn.Conv2d(self.head_dim, self.head_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.head_dim, 1, kernel_size=1),
            nn.Sigmoid()
        )
        self.sh_head = nn.Sequential(
            nn.Conv2d(self.head_dim, self.head_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.head_dim, 3 * self.d_sh, kernel_size=1),
        )

    def forward(self, img, depth, img_feat):
        # img_feat1: [4, 64, 176, 320]
        # img_feat2: [4, 64, 88, 160]
        # img_feat3: [4, 128, 44, 80]
        img_feat1, img_feat2, img_feat3 = img_feat
        # depth_feat1: [4, 32, 176, 320]
        # depth_feat2: [4, 48, 88, 160]
        # depth_feat3: [4, 96, 44, 80]
        depth_feat1, depth_feat2, depth_feat3 = self.depth_encoder(depth)

        feat3 = torch.concat([img_feat3, depth_feat3], dim=1)
        feat2 = torch.concat([img_feat2, depth_feat2], dim=1)
        feat1 = torch.concat([img_feat1, depth_feat1], dim=1)

        up3 = self.decoder3(feat3)
        up3 = F.interpolate(
            up3, size=feat2.shape[-2:], mode="bilinear", align_corners=False)
        up2 = self.decoder2(torch.cat([up3, feat2], dim=1))
        up2 = F.interpolate(
            up2, size=feat1.shape[-2:], mode="bilinear", align_corners=False)
        up1 = self.decoder1(torch.cat([up2, feat1], dim=1))

        up1 = F.interpolate(
            up1, size=img.shape[-2:], mode="bilinear", align_corners=False)
        out = torch.cat([up1, img, depth], dim=1)
        out = self.out_conv(out)
        out = self.out_relu(out)

        # rot head
        rot_out = self.rot_head(out)
        rot_out = torch.nn.functional.normalize(rot_out, dim=1)

        # scale head
        scale_out = torch.clamp_max(self.scale_head(out), 0.01)

        # opacity head
        opacity_out = self.opacity_head(out)

        # sh head
        sh_out = self.sh_head(out)
        # sh_out: [(b * v), C, H, W]

        sh_out = rearrange(
            sh_out, "n c h w -> n (h w) c",
        )
        sh_out = rearrange(
            sh_out,
            "... (srf c) -> ... srf () c",
            srf=1,
        )

        sh_out = rearrange(sh_out, "... (xyz d_sh) -> ... xyz d_sh", xyz=3)
        # [(b * v), (H * W), 1, 1 3, 25]

        # sh_out = sh_out.broadcast_to(sh_out.shape) * self.sh_mask
        sh_out = sh_out * self.sh_mask
        

        return rot_out, scale_out, opacity_out, sh_out
