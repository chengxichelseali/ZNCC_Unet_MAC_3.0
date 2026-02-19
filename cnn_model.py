import torch
import torch.nn as nn


class AdaptiveTanh(nn.Module):
    def __init__(self):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(1))

    def forward(self, x):
        return torch.tanh(self.alpha * x)


class UNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            AdaptiveTanh(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            AdaptiveTanh()
        )

    def forward(self, x):
        return self.block(x)


class UNet(nn.Module):

    def __init__(self, in_channels=3):
        super().__init__()

        self.enc1 = UNetBlock(in_channels, 32)
        self.pool1 = nn.AvgPool2d(2)

        self.enc2 = UNetBlock(32, 64)
        self.pool2 = nn.AvgPool2d(2)

        self.enc3 = UNetBlock(64, 128)
        self.pool3 = nn.AvgPool2d(2)

        self.bottleneck = UNetBlock(128, 256)

        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec3 = UNetBlock(256, 128)

        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec2 = UNetBlock(128, 64)

        self.up1 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.dec1 = UNetBlock(64, 32)

        self.out_conv = nn.Conv2d(32, 5, kernel_size=1) #2变5

    def forward(self, x):
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pool1(enc1))
        enc3 = self.enc3(self.pool2(enc2))

        bottleneck = self.bottleneck(self.pool3(enc3))

        up3 = self.up3(bottleneck)
        dec3 = self.dec3(torch.cat([up3, enc3], dim=1))

        up2 = self.up2(dec3)
        dec2 = self.dec2(torch.cat([up2, enc2], dim=1))

        up1 = self.up1(dec2)
        dec1 = self.dec1(torch.cat([up1, enc1], dim=1))

        out = self.out_conv(dec1)
        return out


DisplacementCNN = UNet
