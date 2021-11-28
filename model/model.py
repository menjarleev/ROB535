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
        input_layer = [nn.Conv2d(input_dim, ngf, kernel_size=7, stride=1, padding=3, padding_mode=padding_mode),
                       nn.BatchNorm2d(ngf),
                       nn.ReLU()]
        self.input = nn.Sequential(*input_layer)
        for i in range(num_res_block):
            res_block = []
            for j in range(max(3 - i, 1)):
                res_block += [RCBAM(ngf, ngf, norm_layer)]
            next_ngf = min(max_channel, ngf * 2)
            res_block += [ResnetBlock(ngf, next_ngf, norm_layer, kernel_size=3, stride=2, padding_mode=padding_mode)]
            res_block = nn.Sequential(*res_block)
            setattr(self, f'res_block_{i}', res_block)
            ngf = next_ngf
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Linear(ngf, num_class)
        self.out = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.input(x)
        bs = x.size(0)
        for i in range(self.num_res_block):
            res_block = getattr(self, f'res_block_{i}')
            x = res_block(x)
        x = self.max_pool(x)
        x = self.fc(x.view(bs, -1))
        x = self.out(x)
        return x