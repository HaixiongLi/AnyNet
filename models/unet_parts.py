""" Parts of the U-Net model """

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class DoubleConv_anynet(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels

        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
        )

    def forward(self, x):
        return self.double_conv(x)

class Preconv2d(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""
    def __init__(self, in_channels, out_channels,kernel_size, stride, pad, dilation=1, bn=True):
        super().__init__()
        if bn:
            self.double_conv = nn.Sequential(
                nn.BatchNorm2d(in_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=dilation if dilation > 1 else pad, dilation=dilation, bias=False),
            )
        else:
            self.double_conv = nn.Sequential(
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=dilation if dilation > 1 else pad, dilation=dilation, bias=False),
            )

    def forward(self, x):
        return self.double_conv(x)


def preconv2d(in_planes, out_planes, kernel_size, stride, pad, dilation=1, bn=True):
    if bn:
        return nn.Sequential(
            nn.BatchNorm2d(in_planes),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=dilation if dilation > 1 else pad, dilation=dilation, bias=False))
    else:
        return nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=dilation if dilation > 1 else pad, dilation=dilation, bias=False))

class Down(nn.Module):
    """Downscaling with maxpool then double conv"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2,2),
            DoubleConv(in_channels, out_channels)
        )
    def forward(self, x):
        return self.maxpool_conv(x)

class Down_anynet(nn.Module):
    """Downscaling with maxpool then double conv"""
    def __init__(self, in_channels, out_channels,nblock=1):
        super().__init__()
        down_block = nn.Sequential()
        down_block.add_module('maxpool',nn.MaxPool2d(2,2))
        for i in range(nblock):
            down_block.add_module('conv'+str(i), preconv2d(in_channels, out_channels, 3, 1, 1))
            in_channels = out_channels
        self.down_block = down_block
    def forward(self, x):
        return self.down_block(x)

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class Up(nn.Module):
    """Upscaling then double conv"""
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels , in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)
    def forward(self, x1, x2):        # input is CHW
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        return self.conv(torch.cat([x2, x1], dim=1))

class unetUp(nn.Module):
    def __init__(self, in_size, out_size, is_deconv):
        super(unetUp, self).__init__()
        self.in_size = in_size
        self.out_size = out_size
        if is_deconv:
            self.up = nn.Sequential(nn.BatchNorm2d(in_size),nn.ReLU(inplace=True),
                nn.ConvTranspose2d(in_size, out_size, kernel_size=2, stride=2, padding=0)
            )
        else:
            self.up = nn.UpsamplingBilinear2d(scale_factor=2)
            in_size = int(in_size * 1.5)
        self.conv = nn.Sequential(preconv2d(in_size, out_size, 3, 1, 1),preconv2d(out_size, out_size, 3, 1, 1),
        )
    def forward(self, x1, x2):
        outputs2 = self.up(x2)
        buttom, right = x1.size(2)%2, x1.size(3)%2
        outputs2 = F.pad(outputs2, (0, -right, 0, -buttom))
        return self.conv(torch.cat([x1, outputs2], dim=1))


class feature_extraction_conv(nn.Module):
    def __init__(self, init_channels,  nblock=2):
        super(feature_extraction_conv, self).__init__()

        self.init_channels = init_channels
        nC = self.init_channels
        downsample_conv = [nn.Conv2d(3,  nC, 3, 1, 1),              #1
                            preconv2d(nC, nC, 3, 2, 1)]             #2
        downsample_conv = nn.Sequential(*downsample_conv)

        inC = nC
        outC = 2*nC
        block0 = self._make_block(inC, outC, nblock)                #3,4,5
        self.block0 = nn.Sequential(downsample_conv, block0)

        nC = 2*nC
        self.blocks = []
        for i in range(2):
            self.blocks.append(self._make_block((2**i)*nC,  (2**(i+1))*nC, nblock)) #6-8,9-11

        self.upblocks = []
        for i in reversed(range(2)):
            self.upblocks.append(unetUp(nC*2**(i+1), nC*2**i, False))   #12-15,16-19

        self.blocks = nn.ModuleList(self.blocks)
        self.upblocks = nn.ModuleList(self.upblocks)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

    def _make_block(self, inC, outC, nblock ):
        model = []
        model.append(nn.MaxPool2d(2,2))
        for i in range(nblock):
            model.append(preconv2d(inC, outC, 3, 1, 1))
            inC = outC
        return nn.Sequential(*model)


    def forward(self, x):
        downs = [self.block0(x)]
        for i in range(2):
            downs.append(self.blocks[i](downs[-1]))
        downs = list(reversed(downs))
        for i in range(1,3):
            downs[i] = self.upblocks[i-1](downs[i], downs[i-1])
        return downs

class shared_feature_extraction_conv(nn.Module):
    def __init__(self, init_channels,  nblock=2):
        super(shared_feature_extraction_conv, self).__init__()

        self.init_channels = init_channels      #1
        nC = self.init_channels

        self.pre_conv = nn.Sequential(nn.Conv2d(3,  nC, 3, 1, 1),
                                      Preconv2d(nC, nC, 3, 2, 1))       #
        i=0
        self.down1 = Down_anynet((2**i)*nC,(2**(i+1))*nC,nblock)        #1/4(nC->2nC)
        i=i+1
        self.down2 = Down_anynet((2**i)*nC,(2**(i+1))*nC,nblock)        #1/8(nC->nC)
        i=i+1
        self.down3 = Down_anynet((2**i)*nC,(2**(i+1))*nC,nblock)        #1/16(4nC->8nC)
        i=i+1
        self.down4 = Down_anynet((2**i)*nC,(2**(i+1))*nC,nblock)        #1/32(8nC->16nC)

        self.up1 = unetUp((2**(i+1))*nC,(2**i)*nC,False)                 #1/16->1/8(16nC->8nC)
        i=i-1
        self.up2 = unetUp((2**(i+1))*nC,(2**i)*nC, False)                #1/8->1/4(8nC->4nC)
        i=i-1
        self.up3 = unetUp((2**(i+1))*nC,(2**i)*nC, False)                #1/4->1/2(4nC->2nC)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        unet_conv = []

        x1 = self.pre_conv(x)   #1/2
        x2 = self.down1(x1)     #1/4
        x3 = self.down2(x2)     #1/8
        x4 = self.down3(x3)     #1/16
        x5 = self.down4(x4)     #1/32
        # unet_conv.append(x5)      #如果想加1/32的feature maps，可以添加进去。
        x = self.up1(x4,x5)
        unet_conv.append(x)     #1/16
        x = self.up2(x3,x)
        unet_conv.append(x)     #1/8
        x = self.up3(x2,x)
        unet_conv.append(x)     #1/4
        # x = self.up4(x1,x)
        # unet_conv.append(x)     #1/2

        return unet_conv


def batch_relu_conv3d(in_planes, out_planes, kernel_size=3, stride=1, pad=1, bn3d=True):
    if bn3d:
        return nn.Sequential(
            nn.BatchNorm3d(in_planes),
            nn.ReLU(),
            nn.Conv3d(in_planes, out_planes, kernel_size=kernel_size, padding=pad, stride=stride, bias=False))
    else:
        return nn.Sequential(
            nn.ReLU(),
            nn.Conv3d(in_planes, out_planes, kernel_size=kernel_size, padding=pad, stride=stride, bias=False))


def post_3dconvs(layers, channels):
    net  = [batch_relu_conv3d(1, channels)]
    net += [batch_relu_conv3d(channels, channels) for _ in range(layers)]
    net += [batch_relu_conv3d(channels, 1)]
    return nn.Sequential(*net)