import torch
import torch.nn as nn
import utils_drunet.basicblock as B


'''
# ====================
# Residual U-Net
# ====================
citation:
@article{zhang2020plug,
title={Plug-and-Play Image Restoration with Deep Denoiser Prior},
author={Zhang, Kai and Li, Yawei and Zuo, Wangmeng and Zhang, Lei and Van Gool, Luc and Timofte, Radu},
journal={arXiv preprint},
year={2020}
}
# ====================
'''
class UNetRes(nn.Module):
    def __init__(self, in_nc=3, out_nc=3, nc=[64, 128, 256], nb=4, act_mode='R', downsample_mode='strideconv', upsample_mode='convtranspose', bias=True, interpolation_mode='bilinear'):
        super(UNetRes, self).__init__()

        self.m_head = B.conv(in_nc, nc[0], bias=bias, mode='C')
        self.interpolation_mode = interpolation_mode

        # downsample
        if downsample_mode == 'avgpool':
            downsample_block = B.downsample_avgpool
        elif downsample_mode == 'maxpool':
            downsample_block = B.downsample_maxpool
        elif downsample_mode == 'strideconv':
            downsample_block = B.downsample_strideconv
        else:
            raise NotImplementedError('downsample mode [{:s}] is not found'.format(downsample_mode))

        self.m_down1 = B.sequential(*[B.ResBlock(nc[0], nc[0], bias=bias, mode='C'+act_mode+'C') for _ in range(nb)], downsample_block(nc[0], nc[1], bias=bias, mode='2'))
        self.m_down2 = B.sequential(*[B.ResBlock(nc[1], nc[1], bias=bias, mode='C'+act_mode+'C') for _ in range(nb)], downsample_block(nc[1], nc[2], bias=bias, mode='2'))

        self.m_body  = B.sequential(*[B.ResBlock(nc[2], nc[2], bias=bias, mode='C'+act_mode+'C') for _ in range(nb)])

        # upsample
        if upsample_mode == 'upconv':
            upsample_block = B.upsample_upconv
        elif upsample_mode == 'pixelshuffle':
            upsample_block = B.upsample_pixelshuffle
        elif upsample_mode == 'convtranspose':
            upsample_block = B.upsample_convtranspose
        else:
            raise NotImplementedError('upsample mode [{:s}] is not found'.format(upsample_mode))

        self.m_up2 = B.sequential(upsample_block(nc[2], nc[1], bias=bias, mode='2'), *[B.ResBlock(nc[1], nc[1], bias=bias, mode='C'+act_mode+'C') for _ in range(nb)])
        self.m_up1 = B.sequential(upsample_block(nc[1], nc[0], bias=bias, mode='2'), *[B.ResBlock(nc[0], nc[0], bias=bias, mode='C'+act_mode+'C') for _ in range(nb)])

        self.m_tail = B.conv(nc[0], out_nc, bias=bias, mode='C')

    def forward(self, x0):
        x1 = self.m_head(x0)
        x2 = self.m_down1(x1)
        x3 = self.m_down2(x2)
        x = self.m_body(x3)

        x = nn.functional.interpolate(x, size=x3.shape[-2:], mode=self.interpolation_mode, align_corners=False)
        x = self.m_up2(x+x3)
        x = nn.functional.interpolate(x, size=x2.shape[-2:], mode=self.interpolation_mode, align_corners=False)
        x = self.m_up1(x+x2)
        x = nn.functional.interpolate(x, size=x1.shape[-2:], mode=self.interpolation_mode, align_corners=False)
        x = self.m_tail(x+x1)

        return x


if __name__ == '__main__':
    x = torch.rand(1,3,256,256)
    net = UNetRes()
    net.eval()
    with torch.no_grad():
        y = net(x)
    print(y.size())
