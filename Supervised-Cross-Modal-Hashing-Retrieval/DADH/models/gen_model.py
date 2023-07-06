import torch
from torch import nn
import torch.nn.init as init
import os
from .CNN_F import image_net


class GEN(torch.nn.Module):
    def __init__(self, dropout, image_dim, text_dim, hidden_dim, output_dim, pretrain_model=None):
        super(GEN, self).__init__()
        self.module_name = 'GEN_module'
        self.output_dim = output_dim
        # self.cnn_f = image_net(pretrain_model)   ## if use 4096-dims feature, pass
        if dropout:
            self.image_module = nn.Sequential(
                nn.Linear(image_dim, hidden_dim, bias=True),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(True),
                nn.Dropout(0.5),
                nn.Linear(hidden_dim, hidden_dim // 2, bias=True),
                nn.BatchNorm1d(hidden_dim // 2),
                nn.ReLU(True),
                nn.Dropout(0.5),
                nn.Linear(hidden_dim // 2, hidden_dim // 4, bias=True),
                nn.BatchNorm1d(hidden_dim // 4),
                nn.ReLU(True),
                nn.Dropout(0.5)
            )
            self.text_module = nn.Sequential(
                nn.Linear(text_dim, hidden_dim, bias=True),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(True),
                nn.Dropout(0.5),
                nn.Linear(hidden_dim, hidden_dim // 2, bias=True),
                nn.BatchNorm1d(hidden_dim // 2),
                nn.ReLU(True),
                nn.Dropout(0.5),
                nn.Linear(hidden_dim // 2, hidden_dim // 4, bias=True),
                nn.BatchNorm1d(hidden_dim // 4),
                nn.ReLU(True),
                nn.Dropout(0.5),
            )
        else:
            self.image_module = nn.Sequential(
                nn.Linear(image_dim, hidden_dim, bias=True),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(True),
                nn.Linear(hidden_dim, hidden_dim // 2, bias=True),
                nn.BatchNorm1d(hidden_dim // 2),
                nn.ReLU(True),
                nn.Linear(hidden_dim // 2, hidden_dim // 4, bias=True),
                nn.BatchNorm1d(hidden_dim // 4),
                nn.ReLU(True),
            )
            self.text_module = nn.Sequential(
                nn.Linear(text_dim, hidden_dim, bias=True),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(True),
                nn.Linear(hidden_dim, hidden_dim // 2, bias=True),
                nn.BatchNorm1d(hidden_dim // 2),
                nn.ReLU(True),
                nn.Linear(hidden_dim // 2, hidden_dim // 4, bias=True),
                nn.BatchNorm1d(hidden_dim // 4),
                nn.ReLU(True)
            )

        self.hash_module = nn.ModuleDict({
            'image': nn.Sequential(
            nn.Linear(hidden_dim // 4, output_dim, bias=True),
            nn.Tanh()),
            'text': nn.Sequential(
            nn.Linear(hidden_dim // 4, output_dim, bias=True),
            nn.Tanh()),
        })


    def weight_init(self):
        initializer = self.kaiming_init
        for block in self._modules:
            if block == 'cnn_f':
                pass
            else:
                for m in self._modules[block]:
                    initializer(m)

    def kaiming_init(self, m):
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            init.kaiming_normal_(m.weight)
            if m.bias is not None:
                m.bias.data.fill_(0)
        elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
            m.weight.data.fill_(1)
            if m.bias is not None:
                m.bias.data.fill_(0)

    def forward(self, x, y):
        # x = self.cnn_f(x).squeeze()   ## if use 4096-dims feature, pass
        f_x = self.image_module(x)
        f_y = self.text_module(y)

        x_code = self.hash_module['image'](f_x).reshape(-1, self.output_dim)
        y_code = self.hash_module['text'](f_y).reshape(-1, self.output_dim)
        return x_code, y_code, f_x.squeeze(), f_y.squeeze()

    def generate_img_code(self, i):
        # i = self.cnn_f(i).squeeze()   ## if use 4096-dims feature, pass
        f_i = self.image_module(i)

        code = self.hash_module['image'](f_i.detach()).reshape(-1, self.output_dim)
        return code

    def generate_txt_code(self, t):
        f_t = self.text_module(t)

        code = self.hash_module['text'](f_t.detach()).reshape(-1, self.output_dim)
        return code

    def load(self, path, use_gpu=False):
        if not use_gpu:
            self.load_state_dict(torch.load(path, map_location=lambda storage, loc: storage))
        else:
            self.load_state_dict(torch.load(path))

    def save(self, name=None, path='./checkpoints', cuda_device=None):
        if not os.path.exists(path):
            os.makedirs(path)
        if cuda_device is not None:
            with torch.cuda.device(cuda_device):
                torch.save(self.state_dict(), os.path.join(path, name))
        else:
            torch.save(self.state_dict(), os.path.join(path, name))
        return name
