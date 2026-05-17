import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.0) -> None:
        super().__init__()
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ]
        if dropout > 0:
            layers.append(nn.Dropout2d(dropout))
        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class UNet(nn.Module):
    def __init__(self, n_channels: int = 3, n_classes: int = 1) -> None:
        super().__init__()

        self.encoder1 = DoubleConv(n_channels, 64)
        self.pool1 = nn.MaxPool2d(2)
        self.encoder2 = DoubleConv(64, 128)
        self.pool2 = nn.MaxPool2d(2)
        self.encoder3 = DoubleConv(128, 256)
        self.pool3 = nn.MaxPool2d(2)
        self.encoder4 = DoubleConv(256, 512, dropout=0.1)
        self.pool4 = nn.MaxPool2d(2)

        self.bottleneck = DoubleConv(512, 1024, dropout=0.2)

        self.up1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.decoder1 = DoubleConv(1024, 512, dropout=0.1)
        self.up2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.decoder2 = DoubleConv(512, 256)
        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.decoder3 = DoubleConv(256, 128)
        self.up4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.decoder4 = DoubleConv(128, 64)

        self.output_layer = nn.Conv2d(64, n_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skip1 = self.encoder1(x)
        skip2 = self.encoder2(self.pool1(skip1))
        skip3 = self.encoder3(self.pool2(skip2))
        skip4 = self.encoder4(self.pool3(skip3))

        x = self.bottleneck(self.pool4(skip4))

        x = self.up1(x)
        x = torch.cat([x, skip4], dim=1)
        x = self.decoder1(x)

        x = self.up2(x)
        x = torch.cat([x, skip3], dim=1)
        x = self.decoder2(x)

        x = self.up3(x)
        x = torch.cat([x, skip2], dim=1)
        x = self.decoder3(x)

        x = self.up4(x)
        x = torch.cat([x, skip1], dim=1)
        x = self.decoder4(x)

        return self.output_layer(x)
