import torch
from torch import nn
import torch.nn.init as init
from torch.nn import ModuleDict
from torch.nn.functional import interpolate
import torchvision
import torch.nn.functional as function
from config import opt


class image_net(nn.Module):
    def __init__(self, pretrain_model):
        super(image_net, self).__init__()
        self.img_module = nn.Sequential(
            # 0 conv1
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=11, stride=(4, 4), padding=(0, 0)),
            # 1 relu1
            nn.ReLU(inplace=True),
            # 2 norm1
            nn.LocalResponseNorm(size=2, k=2),
            # 3 pool1
            nn.ZeroPad2d((0, 1, 0, 1)),
            nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),

            # 4 conv2
            nn.Conv2d(in_channels=64, out_channels=256, kernel_size=5, stride=(1, 1), padding=(2, 2)),
            # 5 relu2
            nn.ReLU(inplace=True),
            # 6 norm2
            nn.LocalResponseNorm(size=2, k=2),
            # 7 pool2
            nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2), padding=(0, 0)),

            # 8 conv3
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=(1, 1), padding=(1, 1)),
            # 9 relu3
            nn.ReLU(inplace=True),

            # 10 conv4
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=(1, 1), padding=(1, 1)),
            # 11 relu4
            nn.ReLU(inplace=True),
            # 12 conv5
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=(1, 1), padding=(1, 1)),
            # 13 relu5
            nn.ReLU(inplace=True),
            # 14 pool5
            nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2), padding=(0, 0)),
            # 15 full_conv6
            nn.Conv2d(in_channels=256, out_channels=4096, kernel_size=6, stride=(1, 1)),
            # 16 relu6
            nn.ReLU(inplace=True),
            # 17 full_conv7
            nn.Conv2d(in_channels=4096, out_channels=4096, kernel_size=1, stride=(1, 1)),
            # 18 relu7
            nn.ReLU(inplace=True)
            # 19 full_conv8
        )
        self.mean = torch.zeros(3, 224, 224)
        self._init(pretrain_model)

    def _init(self, data):
        weights = data['layers'][0]
        self.mean = torch.from_numpy(data['normalization'][0][0][0].transpose()).type(torch.float)
        for i, v in self.img_module.named_children():
            k = int(i)
            if k >= 20:
                break
            if isinstance(v, nn.Conv2d):
                if k > 1:
                    k -= 1
                v.weight.data = torch.from_numpy(weights[k][0][0][0][0][0].transpose())
                v.bias.data = torch.from_numpy(weights[k][0][0][0][0][1].reshape(-1))
        print('sucusses init!')

    def forward(self, x):
        x = x - self.mean.to(opt.device)
        f_x = self.img_module(x)
        return f_x
