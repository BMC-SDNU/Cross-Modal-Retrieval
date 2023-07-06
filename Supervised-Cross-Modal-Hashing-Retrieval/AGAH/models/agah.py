import torch
from torch import nn
from models.basic_module import BasicModule


class AGAH(BasicModule):
    def __init__(self, bit, y_dim, num_label, emb_dim, lambd=0.8, pretrain_model=None):
        super(AGAH, self).__init__()
        self.module_name = 'AGAH'
        self.bit = bit
        self.lambd = lambd

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
            nn.ReLU(inplace=True),

            # 19 full_conv8
            nn.Conv2d(4096, emb_dim, 1),
            nn.ReLU(True)
        )

        self.txt_module = nn.Sequential(
            nn.Conv2d(1, 8192, kernel_size=(y_dim, 1), stride=(1, 1)),
            nn.ReLU(True),
            nn.Conv2d(8192, 4096, kernel_size=1, stride=(1, 1)),
            nn.ReLU(True),
            nn.Conv2d(4096, emb_dim, 1),
            nn.ReLU(True)
        )

        self.hash_module = nn.ModuleDict({
            'img': nn.Sequential(
                nn.Conv2d(emb_dim, bit, 1),
                nn.Tanh()
            ),
            'txt': nn.Sequential(
                nn.Conv2d(emb_dim, bit, 1),
                nn.Tanh()
            )
        })

        self.classifier = nn.ModuleDict({
            'img': nn.Sequential(
                nn.Conv2d(emb_dim, num_label, 1),
                nn.Sigmoid()
            ),
            'txt': nn.Sequential(
                nn.Conv2d(emb_dim, num_label, 1),
                nn.Sigmoid()
            ),
        })

        self.img_discriminator = nn.Sequential(
            nn.Conv2d(1, emb_dim, kernel_size=(emb_dim, 1)),
            nn.ReLU(True),

            nn.Conv2d(emb_dim, 256, 1),
            nn.ReLU(True),

            nn.Conv2d(256, 1, 1)
        )

        self.txt_discriminator = nn.Sequential(
            nn.Conv2d(1, emb_dim, kernel_size=(emb_dim, 1)),
            nn.ReLU(True),

            nn.Conv2d(emb_dim, 256, 1),
            nn.ReLU(True),

            nn.Conv2d(256, 1, 1)
        )

        if pretrain_model is not None:
            self._init(pretrain_model)

    def _init(self, data):
        weights = data['layers'][0]
        for i, v in self.img_module.named_children():
            k = int(i)
            if k >= 20:
                break
            if isinstance(v, nn.Conv2d):
                if k > 1:
                    k -= 1
                v.weight.data = torch.from_numpy(weights[k][0][0][0][0][0].transpose())
                v.bias.data = torch.from_numpy(weights[k][0][0][0][0][1].reshape(-1))

    def forward(self, x, y, feature_map=None):
        f_x = self.img_module(x)
        f_y = self.txt_module(y.unsqueeze(1).unsqueeze(-1))

        # normalization
        f_x = f_x / torch.sqrt(torch.sum(f_x.detach() ** 2))
        f_y = f_y / torch.sqrt(torch.sum(f_y.detach() ** 2))

        # attention
        if feature_map is not None:
            # img attention
            mask_img = torch.sigmoid(5 * f_x.squeeze().mm(feature_map.t()))  # size: (batch, num_label)
            mask_f_x = mask_img.mm(feature_map) / mask_img.sum(dim=1).unsqueeze(-1)  # size: (batch, emb_dim)
            mask_f_x = self.lambd * f_x + (1 - self.lambd) * mask_f_x.unsqueeze(-1).unsqueeze(-1)

            # txt attention
            mask_txt = torch.sigmoid(5 * f_y.squeeze().mm(feature_map.t()))
            mask_f_y = mask_txt.mm(feature_map) / mask_txt.sum(dim=1).unsqueeze(-1)
            mask_f_y = self.lambd * f_y + (1 - self.lambd) * mask_f_y.unsqueeze(-1).unsqueeze(-1)
        else:
            mask_f_x, mask_f_y = f_x, f_y

        x_class = self.classifier['img'](mask_f_x).squeeze()
        y_class = self.classifier['txt'](mask_f_y).squeeze()
        x_code = self.hash_module['img'](mask_f_x).reshape(-1, self.bit)
        y_code = self.hash_module['txt'](mask_f_y).reshape(-1, self.bit)
        return x_code, y_code, f_x.squeeze(), f_y.squeeze(), x_class, y_class

    def dis_img(self, f_x):
        is_img = self.img_discriminator(f_x.unsqueeze(1).unsqueeze(-1))
        return is_img.squeeze()

    def dis_txt(self, f_y):
        is_txt = self.txt_discriminator(f_y.unsqueeze(1).unsqueeze(-1))
        return is_txt.squeeze()

    def generate_img_code(self, x, feature_map=None):
        f_x = self.img_module(x)
        f_x = f_x / torch.sqrt(torch.sum(f_x.detach() ** 2))

        # attention
        if feature_map is not None:
            mask_img = torch.sigmoid(5 * f_x.squeeze().mm(feature_map.t()))  # size: (batch, num_label)
            mask_f_x = mask_img.mm(feature_map) / mask_img.sum(dim=1).unsqueeze(-1)
            f_x = self.lambd * f_x + (1 - self.lambd) * mask_f_x.unsqueeze(-1).unsqueeze(-1)

        code = self.hash_module['img'](f_x).reshape(-1, self.bit)
        return code

    def generate_txt_code(self, y, feature_map=None):
        f_y = self.txt_module(y.unsqueeze(1).unsqueeze(-1))
        f_y = f_y / torch.sqrt(torch.sum(f_y.detach() ** 2))

        # attention
        if feature_map is not None:
            mask_txt = torch.sigmoid(5 * f_y.squeeze().mm(feature_map.t()))
            mask_f_y = mask_txt.mm(feature_map) / mask_txt.sum(dim=1).unsqueeze(-1)
            f_y = self.lambd * f_y + (1 - self.lambd) * mask_f_y.unsqueeze(-1).unsqueeze(-1)

        code = self.hash_module['txt'](f_y).reshape(-1, self.bit)
        return code


