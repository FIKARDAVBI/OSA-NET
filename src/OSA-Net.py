#!/usr/bin/env python

import torch
import torch.nn as nn
import torch.nn.functional as F
from misc.utils import initialize_weights
import pdb

class PreActBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(PreActBlock, self).__init__()
        
        # Initializing normalization and convolution layers
        self.norm1 = nn.BatchNorm2d(in_channels)
        self.norm2 = nn.BatchNorm2d(out_channels)
        self.conv_layer1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.conv_layer2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)

        # Adjust dimensions if stride changes or in_channels != out_channels
        self.adjust = (stride != 1 or in_channels != out_channels)
        if self.adjust:
            self.residual_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)
        
        # Squeeze-and-excitation
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation1 = nn.Conv2d(out_channels, out_channels // 16, kernel_size=1)
        self.excitation2 = nn.Conv2d(out_channels // 16, out_channels, kernel_size=1)

    def forward(self, input_tensor):
        # First stage
        normed_input = self.norm1(input_tensor)
        activated_input = F.relu(normed_input)
        primary_output = self.conv_layer1(activated_input)

        # Second stage
        normed_output = self.norm2(primary_output)
        activated_output = F.relu(normed_output)
        final_conv = self.conv_layer2(activated_output)

        # Squeeze-and-Excitation Mechanism
        scale = self.squeeze(final_conv)
        scale = F.relu(self.excitation1(scale))
        scale = torch.sigmoid(self.excitation2(scale))
        final_conv = final_conv * scale

        # Residual connection
        if self.adjust:
            residual = self.residual_conv(activated_input)
        else:
            residual = input_tensor

        output = final_conv + residual
        return output

class BasicConv(nn.Module):
    def __init__(self, in_channels, out_channels, use_bn=False, **kwargs):
        super(BasicConv, self).__init__()
        self.use_bn = use_bn
        self.conv = nn.Conv2d(in_channels, out_channels, bias=not self.use_bn, **kwargs)
        self.bn = nn.InstanceNorm2d(out_channels, affine=True) if self.use_bn else None

    def forward(self, x):
        x = self.conv(x)
        if self.use_bn:
            x = self.bn(x)
        return F.relu(x, inplace=True)


class BasicDeconv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, use_bn=False):
        super(BasicDeconv, self).__init__()
        self.use_bn = use_bn
        self.tconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride, bias=not self.use_bn)
        self.bn = nn.InstanceNorm2d(out_channels, affine=True) if self.use_bn else None

    def forward(self, x):
        # pdb.set_trace()
        x = self.tconv(x)
        if self.use_bn:
            x = self.bn(x)
        return F.relu(x, inplace=True)


class SAModule_Head(nn.Module):
    def __init__(self, in_channels, out_channels, use_bn):
        super(SAModule_Head, self).__init__()
        branch_out = out_channels // 4
        self.branch1x1 = BasicConv(in_channels, branch_out, use_bn=use_bn,
                            kernel_size=1)
        self.branch3x3 = BasicConv(in_channels, branch_out, use_bn=use_bn,
                            kernel_size=3, padding=1)
        self.branch5x5 = BasicConv(in_channels, branch_out, use_bn=use_bn,
                            kernel_size=5, padding=2)
        self.branch7x7 = BasicConv(in_channels, branch_out, use_bn=use_bn,
                            kernel_size=7, padding=3)
    
    def forward(self, x):
        branch1x1 = self.branch1x1(x)
        branch3x3 = self.branch3x3(x)
        branch5x5 = self.branch5x5(x)
        branch7x7 = self.branch7x7(x)
        out = torch.cat([branch1x1, branch3x3, branch5x5, branch7x7], 1)
        return out


class SAModule(nn.Module):
    def __init__(self, in_channels, out_channels, use_bn):
        super(SAModule, self).__init__()
        branch_out = out_channels // 4
        self.branch1x1 = BasicConv(in_channels, branch_out, use_bn=use_bn,
                            kernel_size=1)
        self.branch3x3 = nn.Sequential(
                        BasicConv(in_channels, 2*branch_out, use_bn=use_bn,
                            kernel_size=1),
                        BasicConv(2*branch_out, branch_out, use_bn=use_bn,
                            kernel_size=3, padding=1),
                        PreActBlock(branch_out,branch_out),
                        )
        self.branch5x5 = nn.Sequential(
                        BasicConv(in_channels, 2*branch_out, use_bn=use_bn,
                            kernel_size=1),
                        BasicConv(2*branch_out, branch_out, use_bn=use_bn,
                            kernel_size=5, padding=2),
                        PreActBlock(branch_out,branch_out),
                        )
        self.branch7x7 = nn.Sequential(
                        BasicConv(in_channels, 2*branch_out, use_bn=use_bn,
                            kernel_size=1),
                        BasicConv(2*branch_out, branch_out, use_bn=use_bn,
                            kernel_size=7, padding=3),
                        PreActBlock(branch_out,branch_out),
                        )
    
    def forward(self, x):
        branch1x1 = self.branch1x1(x)
        branch3x3 = self.branch3x3(x)
        branch5x5 = self.branch5x5(x)
        branch7x7 = self.branch7x7(x)
        out = torch.cat([branch1x1, branch3x3, branch5x5, branch7x7], 1)
        return out


class SANetP1(nn.Module):
    def __init__(self, gray_input=False, use_bn=True):
        super(SANetP1, self).__init__()
        if gray_input:
            in_channels = 1
        else:
            in_channels = 3

        self.encoder = nn.Sequential(
            SAModule_Head(in_channels, 32, use_bn),
            nn.MaxPool2d(2, 2),
            SAModule(32, 64, use_bn),
            nn.MaxPool2d(2, 2),
            SAModule(64, 64, use_bn),
            nn.MaxPool2d(2, 2),
            SAModule(64, 64, use_bn),
            )

        self.decoder = nn.Sequential(
            BasicConv(64, 64, use_bn=use_bn, kernel_size=9, padding=4),
            BasicDeconv(64, 64, 2, stride=2, use_bn=use_bn),
            BasicConv(64, 32, use_bn=use_bn, kernel_size=7, padding=3),
            BasicDeconv(32, 32, 2, stride=2, use_bn=use_bn),
            BasicConv(32, 16,  use_bn=use_bn, kernel_size=5, padding=2),
            BasicDeconv(16, 16, 2, stride=2, use_bn=use_bn),
            BasicConv(16, 16,  use_bn=use_bn, kernel_size=3, padding=1),
            BasicConv(16, 1, use_bn=False, kernel_size=1),
            )

        initialize_weights(self.modules())

    def forward(self, x):
        features = self.encoder(x)
        out = self.decoder(features)
        return out
