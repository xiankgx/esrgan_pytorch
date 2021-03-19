

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast
from torchvision.models import vgg19


class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()

        vgg19_model = vgg19(pretrained=True)
        # self.vgg19_54 = nn.Sequential(
        #     *list(vgg19_model.features.children())[:35])
        self.vgg19_22 = nn.Sequential(
            *list(vgg19_model.features.children())[:8])

    def forward(self, img):
        return self.vgg19_22(img)


class DenseResidualBlock(nn.Module):
    """
    The core module of paper: (Residual Dense Network for Image Super-Resolution, CVPR 18)
    """

    def __init__(self, filters, res_scale=0.2):
        super(DenseResidualBlock, self).__init__()
        self.res_scale = res_scale

        def block(in_features, non_linearity=True):
            layers = [nn.Conv2d(in_features, filters, 3, 1, 1,
                                padding_mode="reflect")]
            if non_linearity:
                layers += [nn.LeakyReLU()]
            return nn.Sequential(*layers)

        self.b1 = block(in_features=1 * filters)
        self.b2 = block(in_features=2 * filters)
        self.b3 = block(in_features=3 * filters)
        self.b4 = block(in_features=4 * filters)
        self.b5 = block(in_features=5 * filters, non_linearity=False)
        self.blocks = [self.b1, self.b2, self.b3, self.b4, self.b5]

    def forward(self, x):
        inputs = x
        for block in self.blocks:
            out = block(inputs)
            inputs = torch.cat([inputs, out], 1)
        return out.mul(self.res_scale) + x


class ResidualInResidualDenseBlock(nn.Module):
    def __init__(self, filters, res_scale=0.2):
        super(ResidualInResidualDenseBlock, self).__init__()
        self.res_scale = res_scale
        self.dense_blocks = nn.Sequential(
            DenseResidualBlock(filters),
            DenseResidualBlock(filters),
            DenseResidualBlock(filters)
        )

    def forward(self, x):
        return self.dense_blocks(x).mul(self.res_scale) + x


class Generator(nn.Module):
    def __init__(self, channels=3, filters=64, num_res_blocks=16, num_upsample=2):
        super(Generator, self).__init__()

        # First layer
        self.conv1 = nn.Conv2d(channels, filters, 3, 1, 1,
                               padding_mode="reflect")

        # Residual blocks
        self.res_blocks = nn.Sequential(*[ResidualInResidualDenseBlock(filters)
                                          for _ in range(num_res_blocks)])

        # Second conv layer post residual blocks
        self.conv2 = nn.Conv2d(filters, filters, 3, 1, 1,
                               padding_mode="reflect")

        # Upsampling layers
        upsample_layers = []
        for _ in range(num_upsample):
            upsample_layers += [
                nn.Conv2d(filters, filters * 4, 3, 1, 1,
                          padding_mode="reflect"),
                nn.LeakyReLU(),
                nn.PixelShuffle(upscale_factor=2),
            ]
        self.upsampling = nn.Sequential(*upsample_layers)

        # Final output block
        self.conv3 = nn.Sequential(
            nn.Conv2d(filters, filters, 3, 1, 1,
                      padding_mode="reflect"),
            nn.LeakyReLU(),
            nn.Conv2d(filters, channels, 3, 1, 1,
                      padding_mode="reflect"),
        )

    def forward(self, x, mixed_precision=False):
        with autocast(mixed_precision):
            out1 = self.conv1(x)
            out = self.res_blocks(out1)
            out2 = self.conv2(out)
            out = torch.add(out1, out2)
            out = self.upsampling(out)
            out = self.conv3(out)
            return out


class Discriminator(nn.Module):
    def __init__(self, in_channels=3):
        super(Discriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, first_block=False):
            layers = []

            layers.append(nn.Conv2d(in_filters, out_filters, 3, 1, 1,
                                    bias=first_block))
            if not first_block:
                layers.append(nn.BatchNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))

            layers.append(nn.Conv2d(out_filters, out_filters, 3, 2, 1,
                                    bias=False))
            layers.append(nn.BatchNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        layers = []
        in_filters = in_channels
        for i, out_filters in enumerate([64, 128, 256, 512]):
            layers.extend(discriminator_block(in_filters, out_filters,
                                              first_block=(i == 0)))
            in_filters = out_filters

        layers.append(nn.Conv2d(out_filters, 1, 3, 1, 1))

        self.model = nn.Sequential(*layers)

    def forward(self, img, mixed_precision=False):
        with autocast(mixed_precision):
            return self.model(img)


if __name__ == "__main__":
    model = FeatureExtractor()
    print(model)
