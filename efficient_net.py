import torch
from torch import nn
import fastai
from fastai.layers import Flatten

def init_cnn(m):
    # Taken from https://github.com/fastai/fastai/blob/master/fastai/vision/models/xresnet.py
    if getattr(m, 'bias', None) is not None: nn.init.constant_(m.bias, 0)
    if isinstance(m, (nn.Conv2d,nn.Linear)): nn.init.kaiming_normal_(m.weight)
    for l in m.children(): init_cnn(l)

class Swish(nn.Module):
    """Swish Activation"""
    def forward(self, x):
        return x * torch.sigmoid(x)
    
class SELayer(nn.Module):
    """Squeeze and Excitation Layer"""
    def __init__(self, nf, sqeeze=0.25):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.sqeeze_expand = nn.Sequential(
            nn.Conv2d(nf, int(nf*sqeeze), 1, stride=1, bias=False),
            Swish(),
            nn.Conv2d(int(nf*sqeeze), nf, 1, stride=1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.sqeeze_expand(y)
        return x * y

def conv_layer(n_in_filters, n_filters, ker_size, stride=1, 
               depthwise=False, zero_bn=False, act=True) :
    """Conv layer followed by batchnorm and possibly Swish activation"""
    bn = nn.BatchNorm2d(n_filters)
    nn.init.constant_(bn.weight, 0. if zero_bn else 1.)
    if act:
        return nn.Sequential(
                nn.Conv2d(n_in_filters, n_filters, ker_size, stride=stride,
                          padding=ker_size//2, bias=False,
                          groups = n_in_filters if depthwise else 1), 
                bn,
                Swish())
    else:
        return nn.Sequential(
                nn.Conv2d(n_in_filters, n_filters, ker_size, stride=stride,
                          padding=ker_size//2,bias=False,
                          groups= n_in_filters if depthwise else 1), 
                bn)

class MBConv(nn.Module):
    """Mobile inverted bottleneck convolutional module with squeeze-and-excitation optimization"""
    def __init__(self, ni, nf, ks, s, exp):
        super().__init__()
        self.ni = ni
        self.nf = nf
        self.exp = exp 
        self.resblock = s == 1 and ni == nf
        if exp != 1:
            self.conv1 = conv_layer(ni, ni*exp, 1)
        self.conv2 = conv_layer(ni*exp, ni*exp, ks, s, depthwise=True)
        self.se = SELayer(ni*exp)
        self.conv3 = conv_layer(ni*exp, nf, 1, zero_bn=self.resblock, act=False)
        
    def forward(self, x):
        y = x
        if self.exp != 1:
            y = self.conv1(y)
        y = self.conv2(y)
        y = self.se(y)
        y = self.conv3(y)
        if self.resblock:
            y = y + x
        return y

class EfficientNetB0(nn.Module):
    """EfficientNetB0 - https://arxiv.org/pdf/1905.11946.pdf
    Example of usage with Imatenet-shaped input images:

    filters = [32, 16, 24, 40, 80, 112, 192, 320, 1280]
    downsample_blocks = [1, 2, 3, 5] Blocks with stride 2 convs
    big_kernel_blocks = [2, 4, 5] # Blocks with 5x5 depthwise conv kernels, others have 3x3
    blocks = [1, 2, 2, 3, 3, 4, 1] # Block lengths

    model = EfficientNetB0(filters, blocks, downsample_blocks, big_kernel_blocks)
    """
    def make_block(self, block_ix, downsample, ker_size, block_len):
        """1 possibly non-residual MBConv block with possible HxW downsampling by factor 's'
         and increasing of depth by factor 2, followed by block_len-1 normal residual MBConvs.
        """
        stride = int(downsample) + 1
        n_in_filters = self.filters[block_ix]
        n_filters = self.filters[block_ix+1]
        mult_fact = 1 if block_ix == 0 else 6

        block = [MBConv(n_in_filters, n_filters, ker_size, stride, mult_fact)]
        block += [MBConv(n_filters, n_filters, ker_size, 1, mult_fact) for _ in range(block_len-1)]
        return block

    def make_head(self, block_ix):
        """End of model: Conv2D - AdaptiveAvgPool2d - Dropout - Linear"""
        n_in_filters = self.filters[block_ix+1]
        n_filters = self.filters[block_ix+2]

        head = [conv_layer(n_in_filters, n_filters, 1), nn.AdaptiveAvgPool2d(1)]
        head += [Flatten(), nn.Dropout(0.2), nn.Linear(n_filters, self.c)]
        return head
    
    def __init__(self, filters, blocks, downsample_blocks, big_kernel_blocks, 
                 n_classes=10):
        super().__init__()
        self.c = n_classes
        self.filters = filters

        layer_blocks = [conv_layer(3, filters[0], 3, 2)]
        for ix,n in enumerate(blocks) :
            ks = 5 if ix in big_kernel_blocks else 3 
            downsample = ix in downsample_blocks
            layer_blocks += self.make_block(ix, downsample, ks, n)
        layer_blocks += self.make_head(ix)

        self.layer_blocks = nn.Sequential(*layer_blocks)
        init_cnn(self)
    
    def forward(self,x) :
        return self.layer_blocks(x) 
