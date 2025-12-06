from torch import nn
import torch
import torch.nn.functional as F
from .networks import get_norm_layer, init_net
from models.attention import EMA
import models.gap as gap
from models.Dsconv import DSConv
from models.WTConv import WTConv2d
from models.FreqFusion import FreqFusion
# from models.rcm import RCM
from models.PKI import InceptionBottleneck
from models.attention import SEBlock
from models.vit import ViT
from einops import rearrange
from models.vmamba_encoder import VSSM_Encoder,VSSLayer,PatchEmbed2D
# 如果你用 notebook，先重启 kernel
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:64"
torch.cuda.empty_cache()


def Conv3X3(in_, out):
    return torch.nn.Conv2d(in_, out, 3, padding=1)


class DWConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels, bias=False)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        
    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x

class ConvRelu(nn.Module):
    def __init__(self, in_, out,norm_layer):
        super().__init__()
        self.conv =nn.Sequential(Conv3X3(in_, out),
                                norm_layer(out)
        )
        self.activation = torch.nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.activation(x)
        return x
    
class ConvReluLite(nn.Module):
    def __init__(self, in_, out,norm_layer):
        super().__init__()
        self.conv =nn.Sequential(DWConv(in_, out),
                                norm_layer(out)
        )
        self.activation = torch.nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.activation(x)
        return x

class WTConvRelu(nn.Module):
    def __init__(self, in_, out,norm_layer):
        super().__init__()
        self.conv1 = nn.Conv2d(in_,out,1)
        self.conv =nn.Sequential(WTConv2d(out, out),
                                norm_layer(out)
        )
        self.activation = torch.nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv(x)
        x = self.activation(x)
        return x

class DSConvRelu(nn.Module):
    def __init__(self, in_, out,norm_layer):
        super().__init__()
        self.conv =nn.Sequential(DSConv(in_, out,3),
                                norm_layer(out)
        )
        self.activation = torch.nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.activation(x)
        return x

class Down(nn.Module):

    def __init__(self, nn):
        super(Down,self).__init__()
        self.nn = nn
        self.maxpool_with_argmax = torch.nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)

    def forward(self,inputs):
        down = self.nn(inputs)
        unpooled_shape = down.size()
        outputs, indices = self.maxpool_with_argmax(down)
        return outputs, down, indices, unpooled_shape
        # return down

class Up(nn.Module):

    def __init__(self, nn):
        super().__init__()
        self.nn = nn
        self.unpool=torch.nn.MaxUnpool2d(2,2)

    def forward(self,inputs,indices,output_shape):
        outputs = self.unpool(inputs, indices=indices, output_size=output_shape)
        outputs = self.nn(outputs)
        return outputs

class Fuse(nn.Module):

    def __init__(self, nn, scale):
        super().__init__()
        self.nn = nn
        self.scale = scale
        self.conv = Conv3X3(64,1)

    def forward(self,down_inp,up_inp,size):
        outputs = torch.cat([down_inp, up_inp], 1)
        outputs = F.interpolate(outputs, size=size, mode='bilinear')
        outputs = self.nn(outputs)

        return self.conv(outputs)

class encoder_Fuse(nn.Module):
    def __init__(self, in_channels, out_channels, norm_layer=nn.BatchNorm2d):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(1, in_channels, 1, 1))
        self.fuse = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            norm_layer(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x1, x2):
        x = self.alpha * x1 + (1 - self.alpha) * x2
        return self.fuse(x)


class SEGuidedAttention(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // reduction, 1),
            nn.ReLU(),
            nn.Conv2d(in_channels // reduction, in_channels, 1),
            nn.Sigmoid()
        )

    def forward(self, feat_main, feat_guide):
        att = self.fc(feat_guide)
        return feat_main * att + feat_guide



class WT_DS_Down(nn.Module):
    def __init__(self, in_channels, out_channels, fuse_mode='concat'):
        super(WT_DS_Down, self).__init__()
        self.conv1 = nn.Conv2d(in_channels,out_channels,1)
        self.wtconv = WTConv2d(out_channels, out_channels)
        self.dsconv = DSConv(out_channels, out_channels,3)
        self.fuse_mode = fuse_mode

        if fuse_mode == 'concat':
            self.fuse_conv = nn.Conv2d(out_channels * 2, out_channels, kernel_size=1)
       

    def forward(self, x):
        x1 = self.conv1(x)
        feat_wt = self.wtconv(x1)
        feat_ds = self.dsconv(x1)

        if self.fuse_mode == 'add':
            fused = feat_wt + feat_ds
        elif self.fuse_mode == 'concat':
            fused = torch.cat([feat_wt, feat_ds], dim=1)
            fused = self.fuse_conv(fused)
        else:
            raise ValueError("Unsupported fuse_mode, use 'add' or 'concat'.")

        return fused


class AttentionFusionDown(nn.Module):
    def __init__(self, wt_ds_block, base_block,out_channels):
        super(AttentionFusionDown, self).__init__()
        self.wt_ds_block = wt_ds_block
        self.base_block = base_block
        self.attn = SEBlock(out_channels*2)
        self.conv1x1 = nn.Conv2d(out_channels*2, out_channels, kernel_size=1)
        self.maxpool_with_argmax = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)

    def forward(self, x):
        f1 = self.wt_ds_block(x)
        f2 = self.base_block(x)

        f_cat = torch.cat([f1, f2], dim=1)
        f_att = self.attn(f_cat)
        fused = self.conv1x1(f_att)

        unpooled_shape = fused.size()
        pooled, indices = self.maxpool_with_argmax(fused)
        return pooled, fused, indices, unpooled_shape



class WT_DS_CascadedGateBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(WT_DS_CascadedGateBlock, self).__init__()

        self.wtconv = WTConv2d(in_channels, out_channels)  # WT模块
        self.dsconv = DSConv(in_channels, out_channels,3)  # 蛇形卷积模块,kernel size =3

        self.fusion_conv = nn.Conv2d(out_channels * 2, out_channels, kernel_size=1)

        # 自适应门控机制，产生一个介于0和1的注意力图
        self.gate_conv = nn.Sequential(
            nn.Conv2d(out_channels * 2, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, 2, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        wt_feat = self.wtconv(x)
        ds_feat = self.dsconv(x)

        # 连接两个特征用于 gate 计算
        concat_feat = torch.cat([wt_feat, ds_feat], dim=1)

        gate = self.gate_conv(concat_feat)  # shape: [B, 2, H, W]
        out = gate[:,0:1] * wt_feat + gate[:,1:2] * ds_feat  # gated fusion

        # Optionally reduce channel count
        out = self.fusion_conv(torch.cat([out, x], dim=1))  # 残差增强

        return out

class SkipWaveletEnhance(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.wtconv = WTConv2d(in_channels, in_channels)
        self.fuse = nn.Conv2d(in_channels * 2, in_channels, 1)

    def forward(self, x):
        x_wt = self.wtconv(x)
        x_cat = torch.cat([x, x_wt], dim=1)
        return self.fuse(x_cat)

class FrenquencyEhanceAttention(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(FrenquencyEhanceAttention, self).__init__()
        self.freq=DCTFrequencyEnhancer(in_channels)
        self.ema = EMA(in_channels)
        self.conv1 = nn.Conv2d(in_channels*2,in_channels,1)
        self.conv3 = nn.Conv2d(in_channels,in_channels,3,padding=1)
        self.fusion = nn.Conv2d(in_channels*2,out_channels,1)

    def forward(self, x):
        freq = self.freq(x)
        # print(freq.shape,'freq')torch.Size([4, 64, 384, 544]) freq
        f1 = torch.cat([freq,x],dim=1)
        f1 = self.conv1(f1)
        f1 = self.conv3(f1)
        ema = self.ema(f1)
        f2 = torch.cat([ema,x],dim=1)
        out = self.fusion(f2)
        out = out+x
        return out



class FEMF(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(FEMF, self).__init__()
        self.ema = EMA(in_channels)
        self.freq = FreqFusion(in_channels, out_channels)

    def forward(self, x_h,x_l):
        e = self.ema(x_h)
        freq = self.freq(e,x_l)
        out = freq+x_h
        return out


class AlignVSSMtoDeepCrack(nn.Module):
    def __init__(self):
        super().__init__()
        # 通道映射层：VSSM通道 → DeepCrack通道
        self.align_convs = nn.ModuleList([
            nn.Conv2d(128, 256, kernel_size=1),   # Layer 1
            nn.Conv2d(256, 512, kernel_size=1),  # Layer 2
            nn.Conv2d(512, 512, kernel_size=1),  # Layer 3
            nn.Conv2d(512, 512, kernel_size=1),  # Layer 4
        ])
        
        # 目标空间大小（H, W）
        self.target_sizes = [
            # (H/4, W/4),   # Layer 1
            # (H/8, W/8),    # Layer 2
            # (H/16, W/16),    # Layer 3
            # (H/16, W/16),    # Layer 4
            (96, 136),   # Layer 1
            (48, 68),    # Layer 2
            (24, 34),    # Layer 3
            (12, 17),    # Layer 4
        ]

    def forward(self, vssm_feats):
        aligned_feats = []
        for i in range(4):
            x = vssm_feats[i]  # x: [B, H, W, C]
            x = x.permute(0, 3, 1, 2)  # 转成 [B, C, H, W]
            x = self.align_convs[i](x)  # 通道映射
            x = F.interpolate(x, size=self.target_sizes[i], mode='bilinear', align_corners=False)  # 尺寸对齐
            aligned_feats.append(x)
        return aligned_feats

class GAU(nn.Module):
    def __init__(self, channels):
        super(GAU, self).__init__()
        self.dwconv3x3 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels)
        self.dwconv5x5 = nn.Conv2d(channels, channels, kernel_size=5, padding=2, groups=channels)
        
        self.fusion_dwconv = nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels)
        self.gelu = nn.GELU()
        self.sigmoid = nn.Sigmoid()
        self.side_conv1x1 = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, x):
        # 主干路径
        dw3 = self.sigmoid(self.dwconv3x3(x))
        dw5 = self.gelu(self.dwconv5x5(x))
        fused = dw3 * dw5                         # 点乘融合
        fused = self.fusion_dwconv(fused)
        fused = self.gelu(fused)

        # 侧分支
        side = self.side_conv1x1(x)

        # 最终输出
        out = fused + side
        return out

class DVM(nn.Module):
    def __init__(self, patch_size=1, in_chans=3, depths=2, 
                 dims=64, d_state=16, drop_rate=0., 
                 attn_drop_rate=0., drop_path_rate=0.1, norm_layer=nn.LayerNorm, 
                 patch_norm=True, use_checkpoint=True):
        super(DVM, self).__init__()
        self.patch_em = PatchEmbed2D(patch_size=patch_size, in_chans=in_chans, embed_dim=dims,
            norm_layer=norm_layer if patch_norm else None)
        self.vss = VSSLayer(dim=dims,# 输入特征的通道数
                depth=depths, # 当前层包含多少个 VSSBlock
                norm_layer=norm_layer,
                downsample=None,
                use_checkpoint=use_checkpoint)

    def forward(self, x):
        x= self.patch_em(x)
        out = self.vss(x)
        out = out+x        # print(out.shape,'mamba vss out')
        # torch.Size([4, 48, 68, 512]) mamba vss out
        # torch.Size([4, 24, 34, 512]) mamba vss out
            # torch.Size([4, 96, 136, 192]) vmamba out shape
        return out

class FFConv(nn.Module):
    def __init__(self, in_, out,norm_layer):
        super(FFConv,self).__init__()
        self.conv = ConvReluLite(in_, out,norm_layer)
        # self.gau = GAU(out)
        self.vss = DVM(in_chans=out,dims=out)
    def forward(self,x):
        x1 = self.conv(x)
        # print(x1.shape,'x1')torch.Size([4, 512, 48, 68]) x1
        # x2= self.gau(x1)
        # print(x2.shape,'x2')torch.Size([4, 512, 48, 68]) x2
        x2 = self.vss(x1)
        x2 = x2.permute(0, 3, 1, 2)  # 转成 [B, C, H, W]
        out = x2+x
        return out
        

class WDcascadeNet(nn.Module):

    def __init__(self, num_classes=1,norm='batch'):
        super(WDcascadeNet, self).__init__()
        norm_layer = get_norm_layer(norm_type=norm)
        self.down1 = Down(torch.nn.Sequential(
            ConvRelu(3,64,norm_layer),
            ConvRelu(64,64,norm_layer),
        ))

        self.down2 = Down(torch.nn.Sequential(
            ConvRelu(64,128,norm_layer),
            ConvRelu(128,128,norm_layer),
            FFConv(128,128,norm_layer),
        ))

        self.down3 = Down(torch.nn.Sequential(
            ConvRelu(128,256,norm_layer),
            ConvRelu(256,256,norm_layer),
            ConvRelu(256,256,norm_layer),
            FFConv(256,256,norm_layer),
        ))

        self.down4 = Down(torch.nn.Sequential(
            ConvRelu(256, 512,norm_layer),
            ConvRelu(512, 512,norm_layer),
            ConvRelu(512, 512,norm_layer),
        ))
        self.down5 = Down(torch.nn.Sequential(
            ConvRelu(512, 512,norm_layer),
            ConvRelu(512, 512,norm_layer),
            ConvRelu(512, 512,norm_layer),
        ))

        # self.vssm_encoder=VSSM_Encoder()
        self.cascade = WT_DS_CascadedGateBlock(512,512)
        
        self.up1 = Up(torch.nn.Sequential(
            ConvRelu(64, 64,norm_layer),
            ConvRelu(64, 64,norm_layer),
        ))

        self.up2 = Up(torch.nn.Sequential(
            ConvRelu(128, 128,norm_layer),
            ConvRelu(128, 64,norm_layer),
        ))

        self.up3 = Up(torch.nn.Sequential(
            ConvRelu(256, 256,norm_layer),
            ConvRelu(256, 256,norm_layer),
            ConvRelu(256, 128,norm_layer),
        ))

        self.up4 = Up(torch.nn.Sequential(
            ConvRelu(512, 512,norm_layer),
            ConvRelu(512, 512,norm_layer),
            ConvRelu(512, 256,norm_layer),
        ))

        self.up5 = Up(torch.nn.Sequential(
            ConvRelu(512, 512,norm_layer),
            ConvRelu(512, 512,norm_layer),
            ConvRelu(512, 512,norm_layer),
        ))

        # self.freq5 = FEMF(512,512)
        # self.freq4 = FEMF(512,512)
        # self.freq3 = FEMF(256,256)
        # self.freq2 = FEMF(128,128)
        # self.freq1 = FEMF(64,64)

        # self.fix_channel1 = nn.Conv2d(512, 256, kernel_size=1)
        # self.fix_channel2 = nn.Conv2d(256, 128, kernel_size=1)
        # self.fix_channel3 = nn.Conv2d(128, 64, kernel_size=1)
        # self.conv2 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=1)
        # self.conv3 = nn.Conv2d(in_channels=256, out_channels=128, kernel_size=1)
        # self.conv4 = nn.Conv2d(in_channels=512, out_channels=256, kernel_size=1)
        # self.conv5 = nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=1)

        # self.side5_conv = nn.Conv2d(512, num_classes, kernel_size=1, stride=1, bias=False)
        # self.side4_conv = nn.Conv2d(256, num_classes, kernel_size=1, stride=1, bias=False)
        # self.side3_conv = nn.Conv2d(128, num_classes, kernel_size=1, stride=1, bias=False)
        # self.side2_conv = nn.Conv2d(64, num_classes, kernel_size=1, stride=1, bias=False)
        # self.final_conv = nn.Conv2d(128, num_classes, kernel_size=1)

        # self.align_convs = AlignVSSMtoDeepCrack()

        # self.encoder_fuse1 = encoder_Fuse(256,256,norm_layer)
        # self.encoder_fuse2 = encoder_Fuse(512,512,norm_layer)
        # self.encoder_fuse3 = encoder_Fuse(512,512,norm_layer)
        # self.encoder_fuse4 = encoder_Fuse(512,512,norm_layer)
        # self.att1 = SEGuidedAttention(256)
        # self.att2 = SEGuidedAttention(512)
        # self.att3 = SEGuidedAttention(512)
        # self.att4 = SEGuidedAttention(512)

        self.fuse5 = Fuse(ConvRelu(512 + 512, 64,norm_layer), scale=16)
        self.fuse4 = Fuse(ConvRelu(512 + 256, 64,norm_layer), scale=8)
        self.fuse3 = Fuse(ConvRelu(256 + 128, 64,norm_layer), scale=4)
        self.fuse2 = Fuse(ConvRelu(128 + 64, 64,norm_layer), scale=2)
        self.fuse1 = Fuse(ConvRelu(64 + 64, 64,norm_layer), scale=1)
        # self.out_fixc = nn.Conv2d(512, 1, kernel_size=1)
        self.final = Conv3X3(5,1)

    def forward(self,inputs):
        h, w = inputs.size()[2:]
        # encoder part
        # 1.vssm_encoder
        # vssm_outs = self.vssm_encoder(inputs)
        # aligned_feats = self.align_convs(vssm_outs)
        # for i in range(len(vssm_outs)):
        #   print(vssm_outs[i].shape,'vssm outputs')
        # torch.Size([4, 96, 136, 192]) vssm outputs
        # torch.Size([4, 48, 68, 384]) vssm outputs
        # torch.Size([4, 24, 34, 768]) vssm outputs
        # torch.Size([4, 24, 34, 768]) vssm outputs
        #2.deepcrack encoder
        out, down1, indices_1, unpool_shape1 = self.down1(inputs)
        # print(down1.shape,"down1")torch.Size([4, 64, 384, 544]) down1
        out, down2, indices_2, unpool_shape2 = self.down2(out)
        out, down3, indices_3, unpool_shape3 = self.down3(out)
        out, down4, indices_4, unpool_shape4 = self.down4(out)
        out, down5, indices_5, unpool_shape5 = self.down5(out)

        # 瓶颈层
        out = self.cascade(out)
        # out = F.interpolate(out,size=down5[2:],mode = 'bilinear',align_corners=True)
        # skip
        # f5 = self.freq5(down5,out)
        # f4 = self.freq4(down4,f5)
        # f4 = self.fix_channel1(f4)
        # # print(f4.shape,'f4 shape')torch.Size([4, 256, 48, 68]) f4 shape
        # f3 = self.freq3(down3,f4)
        # f3 = self.fix_channel2(f3)
        # f2 = self.freq2(down2,f3)
        # f2 = self.fix_channel3(f2)
        # f1 = self.freq1(down1,f2)

        # f1 = self.freqenhance1(down1)
        # f2 = self.freqenhance2(down2)
        # f3 = self.freqenhance3(down3)
        # f4 = self.freqenhance4(down4)
        # f5 = self.freqenhance5(down5)
        # dual encoder fusion
        # print(down3.shape,'d3')
        # print(aligned_feats[0].shape,'align 0')
        # print(out.shape,'out shape')torch.Size([4, 512, 12, 17]) out shape
        # f1 = self.encoder_fuse1(down3,aligned_feats[0])
        # f2 = self.encoder_fuse2(down4,aligned_feats[1])
        # f3 = self.encoder_fuse3(down5,aligned_feats[2])
        # out = self.encoder_fuse4(out,aligned_feats[3])
        #mamba输出作为监督
        # g1 = self.att1(down3,aligned_feats[0])
        # g2 = self.att2(down4,aligned_feats[1])
        # g3 = self.att3(down5,aligned_feats[2])
        # g4 = self.att4(out,aligned_feats[3])
        # 瓶颈层输出作为监督
        # bottle_out = self.out_fixc(out)

        # decoder part
        up5 = self.up5(out, indices=indices_5, output_shape=unpool_shape5)
        # cat_up5 = torch.cat([up5,f5],dim=1)
        # cat_up5 = self.conv5(cat_up5)
        up4 = self.up4(up5, indices=indices_4, output_shape=unpool_shape4)
        # cat_up4 = torch.cat([up4,f4],1)
        # cat_up4 = self.conv4(cat_up4)
        up3 = self.up3(up4, indices=indices_3, output_shape=unpool_shape3)
        # cat_up3 = torch.cat([up3,f3],1)
        # cat_up3 = self.conv3(cat_up3)
        up2 = self.up2(up3, indices=indices_2, output_shape=unpool_shape2)
        # cat_up2 = torch.cat([up2,f2],1)
        # cat_up2 = self.conv2(cat_up2)
        up1 = self.up1(up2, indices=indices_1, output_shape=unpool_shape1)
        # cat_up1 = torch.cat([up1,f1],1)


        # cat_up1 = self.conv1(cat_up1)
        # print(concat_up1.shape,'concat1')torch.Size([4, 64, 256, 256]) concat1
        # side_output5 = self.side5_conv(cat_up5)
        # side_output4 = self.side4_conv(cat_up4)
        # side_output3 = self.side3_conv(cat_up3)
        # side_output2 = self.side2_conv(cat_up2)
        # side_output1 = self.final_conv(cat_up1)

        # side_output2 = F.interpolate(side_output2, size=(h, w), mode='bilinear',
        #                              align_corners=True)  # self.up2(side_output2)
        # side_output3 = F.interpolate(side_output3, size=(h, w), mode='bilinear',
        #                              align_corners=True)  # self.up4(side_output3)
        # side_output4 = F.interpolate(side_output4, size=(h, w), mode='bilinear',
        #                              align_corners=True)  # self.up8(side_output4)
        # side_output5 = F.interpolate(side_output5, size=(h, w), mode='bilinear',
        #                              align_corners=True)  # self.up16(side_output5)
        # fused = self.fuse_conv(torch.cat([side_output1,
        #                                   side_output2,
        #                                   side_output3,
        #                                   side_output4,
        #                                   side_output5], dim=1))
        fuse5 = self.fuse5(down_inp=down5,up_inp=up5,size=[inputs.shape[2],inputs.shape[3]])
        fuse4 = self.fuse4(down_inp=down4, up_inp=up4,size=[inputs.shape[2],inputs.shape[3]])
        fuse3 = self.fuse3(down_inp=down3, up_inp=up3,size=[inputs.shape[2],inputs.shape[3]])
        fuse2 = self.fuse2(down_inp=down2, up_inp=up2,size=[inputs.shape[2],inputs.shape[3]])
        fuse1 = self.fuse1(down_inp=down1, up_inp=up1,size=[inputs.shape[2],inputs.shape[3]])
        output = self.final(torch.cat([fuse5,fuse4,fuse3,fuse2,fuse1],1))

        # side_output5 = self.side5_conv(cat_up5)
        # side_output4 = self.side4_conv(cat_up4)
        # side_output3 = self.side3_conv(cat_up3)
        # side_output2 = self.side2_conv(cat_up2)
        # final_out = self.final_conv(up1)

        # side_output2 = F.interpolate(side_output2, size=(h, w), mode='bilinear',
        #                              align_corners=True)  # self.up2(side_output2)
        # side_output3 = F.interpolate(side_output3, size=(h, w), mode='bilinear',
        #                              align_corners=True)  # self.up4(side_output3)
        # side_output4 = F.interpolate(side_output4, size=(h, w), mode='bilinear',
        #                              align_corners=True)  # self.up8(side_output4)
        # side_output5 = F.interpolate(side_output5, size=(h, w), mode='bilinear',
                                    #  align_corners=True)  # self.up16(side_output5)
        # output = self.final(torch.cat([fuse5,fuse4,fuse3,fuse2,fuse1],1))

        return fuse1, fuse2, fuse3, fuse4, fuse5,output
        # return  side_output1, side_output2, side_output3, side_output4, side_output5
        # # return output
        # return  fuse1, fuse2, fuse3, output

def define_WDcascadeNet(in_nc, 
                     num_classes, 
                     ngf, 
                     norm='batch',
                     init_type='xavier', 
                     init_gain=0.02, 
                     gpu_ids=[]):
    net = WDcascadeNet(num_classes, norm)
    return init_net(net, init_type, init_gain, gpu_ids)

class BinaryFocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, logits=False, size_average=True):
        super(BinaryFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.logits = logits
        self.size_average = size_average
        self.criterion = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, inputs, targets):
        targets=targets.float()
        BCE_loss = self.criterion(inputs, targets)
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss

        if self.size_average:
            return F_loss.mean()
        else:
            return F_loss.sum()



 
class dice_bce_loss(nn.Module):
    def __init__(self, batch=True):
        super(dice_bce_loss, self).__init__()
        self.batch = batch
        self.bce_loss = nn.BCEWithLogitsLoss()
 
    def soft_dice_coeff(self, y_pred,y_true):
        smooth = 1e-6  # may change
        if self.batch:
            i = torch.sum(y_true)
            j = torch.sum(y_pred)
            intersection = torch.sum(y_true * y_pred)
        else:
            i = y_true.sum(1).sum(1).sum(1)
            j = y_pred.sum(1).sum(1).sum(1)
            intersection = (y_true * y_pred).sum(1).sum(1).sum(1)
        score = (2. * intersection + smooth) / (i + j + smooth)
        # score = (intersection + smooth) / (i + j - intersection + smooth)#iou
        return score.mean()
 
    def soft_dice_loss(self, y_true, y_pred):
        loss = 1 - self.soft_dice_coeff(y_true, y_pred)
        return loss
 
    def forward(self, y_pred ,y_true):
        y_true= y_true.float()
        a = self.bce_loss(y_pred, y_true)
        b = self.soft_dice_loss(y_pred,y_true)
        return 0.8*a + 0.2*b
 