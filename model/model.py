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
        input_layer = [nn.Conv2d(input_dim, ngf, kernel_size=3, stride=2, padding=1, padding_mode=padding_mode),
                       nn.BatchNorm2d(ngf),
                       nn.ReLU(inplace=True)]
        self.input = nn.Sequential(*input_layer)
        for i in range(num_res_block):
            res_block = []
            res_block += [RCBAM(ngf, ngf, norm_layer, kernel_size=1),
                          RCBAM(ngf, ngf, norm_layer, kernel_size=3),
                          RCBAM(ngf, ngf, norm_layer, kernel_size=1)]
            next_ngf = min(max_channel, ngf * 2)
            res_block += [ResnetBlock(ngf, next_ngf, norm_layer, kernel_size=3, stride=2, padding_mode=padding_mode)]
            res_block = nn.Sequential(*res_block)
            setattr(self, f'res_block_{i}', res_block)
            ngf = next_ngf
        self.agg = nn.Sequential(SpatialPyramidPooling(ngf, ngf),
                                 SpatialPyramidPooling(ngf, ngf),
                                 SpatialPyramidPooling(ngf, ngf))
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
        x = self.agg(x)
        x = self.max_pool(x)
        x = self.fc2(self.actv(self.fc1(x.view(bs, -1))))
        return x