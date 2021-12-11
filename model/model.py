from .module import *
from torch import nn


class PureRGBNet(nn.Module):
    def __init__(self,
                 num_res_block,
                 ngf,
                 max_channel,
                 input_dim,
                 num_class,
                 norm_layer=nn.BatchNorm2d,
                 padding_mode='reflect'):
        super(PureRGBNet, self).__init__()
        self.num_res_block = num_res_block
        input_layer = [nn.Conv2d(input_dim, ngf, kernel_size=7, stride=2, padding=1, padding_mode=padding_mode),
                       nn.BatchNorm2d(ngf),
                       nn.ReLU(inplace=True)]
        self.input = nn.Sequential(*input_layer)
        for i in range(num_res_block):
            res_block = []
            res_block += [RCBAM(ngf, ngf, norm_layer, kernel_size=3),
                          RCBAM(ngf, ngf, norm_layer, kernel_size=3),
                          RCBAM(ngf, ngf, norm_layer, kernel_size=3)]
            next_ngf = min(max_channel, ngf * 2)
            res_block += [ResnetBlock(ngf, next_ngf, norm_layer, kernel_size=3, stride=2, padding_mode=padding_mode)]
            res_block = nn.Sequential(*res_block)
            setattr(self, f'res_block_{i}', res_block)
            ngf = next_ngf
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Linear(ngf, 1024)
        self.actv = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(1024, num_class)

    def forward(self, x):
        x = self.input(x)
        bs = x.size(0)
        for i in range(self.num_res_block):
            res_block = getattr(self, f'res_block_{i}')
            x = res_block(x)
        x = self.max_pool(x)
        x = self.fc2(self.actv(self.fc1(x.view(bs, -1))))
        return x


class ModifiedFPN(nn.Module):
    def __init__(self,
                 num_res_block,
                 ngf,
                 max_channel,
                 input_dim,
                 num_class,
                 norm_layer=nn.BatchNorm2d,
                 padding_mode='reflect',
                 scale=3):
        super(ModifiedFPN, self).__init__()
        self.num_res_block = num_res_block
        input_layer = [nn.Conv2d(input_dim, ngf, kernel_size=7, stride=2, padding=1, padding_mode=padding_mode),
                       nn.BatchNorm2d(ngf),
                       nn.ReLU(inplace=True)]
        self.input = nn.Sequential(*input_layer)
        out_channel = min(ngf * (2 ** (num_res_block - 1)), max_channel)

        for i in range(num_res_block):
            res_block = []
            res_block += [RCBAM(ngf, ngf, norm_layer, kernel_size=3),
                          RCBAM(ngf, ngf, norm_layer, kernel_size=3),
                          RCBAM(ngf, ngf, norm_layer, kernel_size=3)]
            next_ngf = min(max_channel, ngf * 2)
            res_block += [ResnetBlock(ngf, next_ngf, norm_layer, kernel_size=3, stride=2, padding_mode=padding_mode)]
            res_block = nn.Sequential(*res_block)
            setattr(self, f'res_block_{i}', res_block)
            if i + scale >= num_res_block:
                feat_layer = nn.Sequential(nn.Conv2d(next_ngf, out_channel, 1, 1, 0),
                                           nn.BatchNorm2d(out_channel),
                                           nn.ReLU(True))
                setattr(self, f'feat_layer_{i}', feat_layer)
            ngf = next_ngf
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        for i in range(scale):
            out_layer = nn.Sequential(
                nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=1, padding=1, padding_mode=padding_mode),
                nn.BatchNorm2d(out_channel),
                nn.ReLU(True))

            fc_layer = nn.Sequential(
                nn.Linear(out_channel, 512),
                nn.ReLU(True),
                nn.Linear(512, num_class)
            )
            setattr(self, f'out_layer_{i}', out_layer)
            setattr(self, f'fc_layer_{i}', fc_layer)
        self.scale = scale

    def forward(self, x):
        x = self.input(x)
        bs = x.size(0)
        out = []
        for i in range(self.num_res_block):
            res_block = getattr(self, f'res_block_{i}')
            x = res_block(x)
            if i + self.scale >= self.num_res_block:
                feat_layer = getattr(self, f'feat_layer_{i}')
                y = feat_layer(x)
                out += [y]
        out = out[::-1]
        pred = []
        for i in range(0, self.scale):
            if i == 0:
                x = out[0]
            else:
                x = F.interpolate(x, scale_factor=2)
                x = x + out[i]
            out_layer = getattr(self, f'out_layer_{i}')
            fc_layer = getattr(self, f'fc_layer_{i}')
            y = out_layer(x)
            y = fc_layer(self.max_pool(y).view(bs, -1))
            pred += [y]
        return pred
