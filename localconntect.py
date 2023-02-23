from torch.nn.modules.utils import _pair
import torch
import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
import torch.nn.utils.weight_norm as weight_norm
import torch.nn.functional as F
## Tree-connected https://github.com/OliverRichter/TreeConnect
### https://discuss.pytorch.org/t/locally-connected-layers/26979/3

class LocallyConnected2d(nn.Module):
    def __init__(self, in_channels, out_channels, output_size, kernel_size, stride, bias=False):
        super(LocallyConnected2d, self).__init__()
        output_size = _pair(output_size)
        self.weight = nn.Parameter(
            torch.randn(1, out_channels, in_channels, output_size[0], output_size[1], kernel_size**2)
        )
        if bias:
            self.bias = nn.Parameter(
                torch.randn(1, out_channels, output_size[0], output_size[1])
            )
        else:
            self.register_parameter('bias', None)
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        
    def forward(self, x):
        _, c, h, w = x.size()
        kh, kw = self.kernel_size
        dh, dw = self.stride
        x = x.unfold(2, kh, dh).unfold(3, kw, dw)
        x = x.contiguous().view(*x.size()[:-2], -1)
        # Sum in in_channel and kernel_size dims
        x = x.unsqueeze(1) 
        out = (x * self.weight)
        out = out.sum([2, -1])
        if self.bias is not None:
            out += self.bias
        return out

class LocallyConnected1d(nn.Module):
    def __init__(self, in_channels, out_channels, output_size, kernel_size, stride, bias=False):
        super(LocallyConnected1d, self).__init__()
        self.weight = nn.Parameter(
            torch.randn(1, out_channels, in_channels, output_size, kernel_size)
        )
        if bias:
            self.bias = nn.Parameter(
                torch.randn(1, out_channels, output_size)
            )
        else:
            self.register_parameter('bias', None)
        self.kernel_size = kernel_size
        self.stride = stride
        
    def forward(self, x):
        _, c, h = x.size()
        kh = self.kernel_size
        dh = self.stride
        x = x.unfold(2, kh, dh)
        # x = x.contiguous().view(*x.size()[:-2], -1)
        # Sum in in_channel and kernel_size dims
        x = x.unsqueeze(1)
        out = (x * self.weight)
        out = out.sum([2, -1])
        if self.bias is not None:
            out += self.bias
        return out

