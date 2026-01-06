import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class UpConv(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv = nn.Conv2d(in_c, out_c, kernel_size=3, padding=1)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.up(x)
        x = self.conv(x)
        return self.act(x)


class Model(nn.Module):

    def __init__(self, base_model, out_dim, img_res=(256, 256)):
        super().__init__()
        self.img_res = img_res

        if base_model == "resnet18":
            self.encoder = models.resnet18(weights=False)
            enc_out_dim = 512
        elif base_model == "resnet50":
            self.encoder = models.resnet50(weights=False)
            enc_out_dim = 2048
        else:
            raise ValueError("Invalid backbone")

        feat_dim = 128
        self.encoder.fc = nn.Linear(enc_out_dim, feat_dim)

        self.log_sigma_align = nn.Parameter(torch.zeros(1))
        self.log_sigma_recon = nn.Parameter(torch.zeros(1))

        self.projector = nn.Sequential(
            nn.Linear(feat_dim, feat_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feat_dim, out_dim)
        )

        self.decoder_fc = nn.Sequential(
            nn.Linear(feat_dim, 256 * 4 * 4),
            nn.ReLU(inplace=True)
        )

        self.decoder = nn.Sequential(
            UpConv(256, 128),   # 4 -> 8
            UpConv(128, 64),    # 8 -> 16
            UpConv(64, 32),     # 16 -> 32
            UpConv(32, 16),     # 32 -> 64
            UpConv(16, 8),      # 64 -> 128
            nn.Conv2d(8, 1, kernel_size=3, padding=1)
        )
        
    def forward(self, x):
        feat = self.encoder(x)      # (B, feat_dim)
        z = self.projector(feat)    # (B, out_dim)
        return z

    def forward_with_depth(self, x):
        d = self.decoder_fc(x)
        d = d.view(-1, 256, 4, 4)

        depth = self.decoder(d)   # (B, 1, ~128, ~128)

        depth = F.interpolate(
            depth,
            size=self.img_res,
            mode="bilinear",
            align_corners=False
        )

        return depth
