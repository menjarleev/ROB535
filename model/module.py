import torch as th
from torch import nn
from functools import partial
import torch.nn.functional as F

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=4):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = th.mean(x, dim=1, keepdim=True)
        max_out, _ = th.max(x, dim=1, keepdim=True)
        x = th.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class CBAM(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size=3, stride=1, padding_mode='reflect'):
        super(CBAM, self).__init__()
        self.conv = nn.Conv2d(in_channel, out_channel, kernel_size=kernel_size, stride=stride, padding=kernel_size // 2, padding_mode=padding_mode)
        self.channel_attention = ChannelAttention(out_channel)
        self.spatial_attention = SpatialAttention()

    def forward(self, x):
        x = self.conv(x)
        x = self.channel_attention(x) * x
        x = self.spatial_attention(x) * x
        return x

class RCBAM(nn.Module):
    def __init__(self, in_channel, out_channel, norm_layer, kernel_size=3, stride=1, padding_mode='reflect'):
        super(RCBAM, self).__init__()
        self.conv1 = nn.Conv2d(in_channel, out_channel, kernel_size=kernel_size, stride=stride, padding=kernel_size // 2, padding_mode=padding_mode)
        self.channel1 = ChannelAttention(out_channel)
        self.spatial1 = SpatialAttention()
        self.norm1 = norm_layer(out_channel)
        self.act = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channel, out_channel, kernel_size=kernel_size, stride=stride, padding=kernel_size // 2, padding_mode=padding_mode)
        self.channel2 = ChannelAttention(out_channel)
        self.spatial2 = SpatialAttention()
        self.norm2 = norm_layer(out_channel)

    def forward(self, feature):
        x = self.conv1(feature)
        x = self.channel1(x) * x
        x = self.spatial1(x) * x
        x = self.act(self.norm1(x))
        x = self.conv2(x)
        x = self.channel1(x) * x
        x = self.spatial1(x) * x
        x = self.norm2(x)
        return x + feature

class ResnetBlock(nn.Module):
    def __init__(self, in_channel, out_channel,  norm_layer, kernel_size=3, stride=1, dropout=.0, padding_mode='reflect'):
        super(ResnetBlock, self).__init__()
        self.downscale = None
        if stride > 1:
            self.downscale = nn.Conv2d(in_channel, out_channel, kernel_size=kernel_size, stride=stride, padding=kernel_size // 2, padding_mode=padding_mode)
        self.res_block = self._build_conv_block(in_channel=in_channel,
                                                out_channel=out_channel,
                                                padding_mode=padding_mode,
                                                norm_layer=norm_layer,
                                                kernel_size=kernel_size,
                                                stride=stride,
                                                dropout=dropout)


    def _build_conv_block(self, in_channel, out_channel, padding_mode, norm_layer, kernel_size=3, stride=2, dropout=0.):
        conv_block = []
        conv_block += [nn.Conv2d(in_channel, in_channel, kernel_size=kernel_size, padding=kernel_size // 2, padding_mode=padding_mode),
                       norm_layer(in_channel),
                       nn.ReLU(inplace=True)]
        if dropout > 0.:
            conv_block += [nn.Dropout(dropout)]
        conv_block += [nn.Conv2d(in_channel, out_channel, kernel_size=3, padding=kernel_size // 2, padding_mode=padding_mode, stride=stride),
                       norm_layer(out_channel)]
        return nn.Sequential(*conv_block)

    def forward(self, x):
        scale_x = x
        if self.downscale is not None:
            scale_x = self.downscale(scale_x)
        out = scale_x + self.res_block(x)
        return out

class SpatialPyramidPooling(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(SpatialPyramidPooling, self).__init__()
        self.conv1 = nn.Conv2d(in_channel, in_channel, kernel_size=3, stride=1, padding=1, padding_mode='reflect')
        self.conv2 = nn.Sequential(nn.Conv2d(in_channel, in_channel // 4, kernel_size=3, stride=1, padding=1, padding_mode='reflect'),
                                   nn.ReLU(True),
                                   nn.Conv2d(in_channel // 4, in_channel, kernel_size=3, stride=1, padding=1, padding_mode='reflect'))

        self.conv3 = nn.Sequential(nn.Conv2d(in_channel, in_channel // 4, kernel_size=3, stride=1, padding=1, padding_mode='reflect'),
                                   nn.ReLU(True),
                                   nn.Conv2d(in_channel // 4, in_channel // 4, kernel_size=3, stride=1, padding=1, padding_mode='reflect'),
                                   nn.ReLU(True),
                                   nn.Conv2d(in_channel // 4, in_channel, kernel_size=3, stride=1, padding=1, padding_mode='reflect'))
        self.actv = nn.ReLU(True)
        self.agg = nn.Conv2d(3 * in_channel, out_channel, kernel_size=1)
        self.reduce = None
        if in_channel != out_channel:
            self.reduce = nn.Conv2d(in_channel, out_channel, kernel_size=1)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x3 = self.conv3(x)
        out = th.cat([x1, x2, x3], dim=1)
        out = self.agg(self.actv(out))
        if self.reduce:
            x = self.reduce(x)
        return out + x
