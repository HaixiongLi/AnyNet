
""" Full assembly of the parts to form the complete network """

import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from .unet_parts import *
from .unet_parts import post_3dconvs,shared_feature_extraction_conv,feature_extraction_conv
import sys
from torch.utils.tensorboard import SummaryWriter



class shared_UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(shared_UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.nblocks = 2  # 2
        self.init_channels = 64  # 64,Unet 结构的网络的channe数量
        self.scale = [8*self.n_channels,4*self.n_channels,2*self.n_channels]

        self.feature_extraction = shared_feature_extraction_conv(self.init_channels,self.nblocks)
        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.deal1 = OutConv(8 * self.init_channels, self.n_classes)
        self.deal2 = OutConv(4 * self.init_channels, self.n_classes)
        self.deal3 = OutConv(2 * self.init_channels, self.n_classes)

        self.outc = OutConv(64, n_classes)  #TODO 需要修改

    def forward(self, img):
        feats = self.feature_extraction(img)

        out_tmp1 = self.deal1(feats[0])
        out_tmp2 = self.deal2(feats[1]) + self.up1(out_tmp1)
        logits = self.deal3(feats[2]) + self.up2(out_tmp2)

        # logits = self.outc(x)
        return logits



class SharedNet(nn.Module):
    def __init__(self,args,bilinear=True):
        super(SharedNet,self).__init__()
        self.n_channels = args.img_channels
        self.n_classes = args.num_classes
        self.bilinear = bilinear
        self.nblocks = 2  # 2
        self.init_channels = 64  # 64,Unet 结构的网络的channe数量
        self.maxdisplist = args.maxdisplist         #[12, 3, 3]
        self.spn_init_channels = args.spn_init_channels     #8
        self.nblocks = args.nblocks     #2
        self.layers_3d = args.layers_3d     #4
        self.channels_3d = args.channels_3d     #4
        self.growth_rate = args.growth_rate     #[4,1,1]
        self.with_spn = args.with_spn       #yes
        self.scale = [8*self.n_channels,4*self.n_channels,2*self.n_channels]

        self.feature_extraction = shared_feature_extraction_conv(self.init_channels,self.nblocks)
        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.deal1 = OutConv(8 * self.init_channels, self.n_classes)
        self.deal2 = OutConv(4 * self.init_channels, self.n_classes)
        self.deal3 = OutConv(2 * self.init_channels, self.n_classes)

    def warp(self, x, disp):        #根据img2和视差disp估计恢复img1
        """
        warp an image/tensor (im2) back to im1, according to the optical flow
        x: [B, C, H, W] (im2)
        flo: [B, 2, H, W] flow
        """
        B, C, H, W = x.size()
        # mesh grid
        xx = torch.arange(0, W, device='cuda').view(1, -1).repeat(H, 1)
        yy = torch.arange(0, H, device='cuda').view(-1, 1).repeat(1, W)
        xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1)
        yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1)
        vgrid = torch.cat((xx, yy), 1).float()

        # vgrid = Variable(grid)
        vgrid[:,:1,:,:] = vgrid[:,:1,:,:] - disp

        # scale grid to [-1,1]
        vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :].clone() / max(W - 1, 1) - 1.0
        vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :].clone() / max(H - 1, 1) - 1.0

        vgrid = vgrid.permute(0, 2, 3, 1)
        output = nn.functional.grid_sample(x, vgrid)
        return output


    def _build_volume_2d(self, feat_l, feat_r, maxdisp, stride=1):
        assert maxdisp % stride == 0  # Assume maxdisp is multiple of stride
        cost = torch.zeros((feat_l.size()[0], maxdisp//stride, feat_l.size()[2], feat_l.size()[3]), device='cuda')
        for i in range(0, maxdisp, stride):
            cost[:, i//stride, :, :i] = feat_l[:, :, :, :i].abs().sum(1)
            if i > 0:
                cost[:, i//stride, :, i:] = torch.norm(feat_l[:, :, :, i:] - feat_r[:, :, :, :-i], 1, 1)    #按一维度求一范数
            else:
                cost[:, i//stride, :, i:] = torch.norm(feat_l[:, :, :, :] - feat_r[:, :, :, :], 1, 1)

        return cost.contiguous()

    def _build_volume_2d3(self, feat_l, feat_r, maxdisp, disp, stride=1):       #TODO ?
        size = feat_l.size()
        batch_disp = disp[:,None,:,:,:].repeat(1, maxdisp*2-1, 1, 1, 1).view(-1,1,size[-2], size[-1])
        batch_shift = torch.arange(-maxdisp+1, maxdisp, device='cuda').repeat(size[0])[:,None,None,None] * stride
        batch_disp = batch_disp - batch_shift.float()
        batch_feat_l = feat_l[:,None,:,:,:].repeat(1,maxdisp*2-1, 1, 1, 1).view(-1,size[-3],size[-2], size[-1])
        batch_feat_r = feat_r[:,None,:,:,:].repeat(1,maxdisp*2-1, 1, 1, 1).view(-1,size[-3],size[-2], size[-1])
        cost = torch.norm(batch_feat_l - self.warp(batch_feat_r, batch_disp), 1, 1)
        cost = cost.view(size[0],-1, size[2],size[3])
        return cost.contiguous()

    def forward(self,left,right):
        img_size = left.size()
        feats_l = self.feature_extraction(left)
        feats_r = self.feature_extraction(right)

        sem_tmp1 = self.deal1(feats_l[0])
        sem_tmp2 = self.deal2(feats_l[1]) + self.up1(sem_tmp1)
        sem_res = self.deal3(feats_l[2]) + self.up2(sem_tmp2)

        pred = []
        for scale in range(len(feats_l)):
            if scale > 0:
                wflow = F.upsample(pred[scale-1], (feats_l[scale].size(2), feats_l[scale].size(3)),
                                   mode='bilinear') * feats_l[scale].size(2) / img_size[2]
                cost = self._build_volume_2d3(feats_l[scale], feats_r[scale],
                                         self.maxdisplist[scale], wflow, stride=1)
            else:
                cost = self._build_volume_2d(feats_l[scale], feats_r[scale],
                                             self.maxdisplist[scale], stride=1)

            cost = torch.unsqueeze(cost, 1)
            cost = self.volume_postprocess[scale](cost)
            cost = cost.squeeze(1)
            if scale == 0:
                pred_low_res = disparityregression2(0, self.maxdisplist[0])(F.softmax(-cost, dim=1))
                pred_low_res = pred_low_res * img_size[2] / pred_low_res.size(2)
                disp_up = F.upsample(pred_low_res, (img_size[2], img_size[3]), mode='bilinear')
                pred.append(disp_up)
            else:
                pred_low_res = disparityregression2(-self.maxdisplist[scale]+1, self.maxdisplist[scale], stride=1)(F.softmax(-cost, dim=1))
                pred_low_res = pred_low_res * img_size[2] / pred_low_res.size(2)
                disp_up = F.upsample(pred_low_res, (img_size[2], img_size[3]), mode='bilinear')
                pred.append(disp_up+pred[scale-1])


        return sem_res, pred



class disparityregression2(nn.Module):
    def __init__(self, start, end, stride=1):
        super(disparityregression2, self).__init__()
        self.disp = torch.arange(start*stride, end*stride, stride, device='cuda', requires_grad=False).view(1, -1, 1, 1).float()

    def forward(self, x):
        disp = self.disp.repeat(x.size()[0], 1, x.size()[2], x.size()[3])
        out = torch.sum(x * disp, 1, keepdim=True)
        return out