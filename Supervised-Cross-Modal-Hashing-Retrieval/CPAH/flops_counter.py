import torchvision.models as models
import torch
from ptflops import get_model_complexity_info
from thop import profile
from torch import nn
from config import opt


def test_ptflops():
    net = models.resnet50()
    macs, params = get_model_complexity_info(net, (3, 224, 224), as_strings=False, print_per_layer_stat=False)
    print('{:<40}  {:<8}'.format('Computational complexity (MACs):', macs))
    print('{:<40}  {:<8}'.format('Number of parameters:', params))


def test_thop():
    model = models.resnet50()
    input = torch.randn(1, 3, 224, 224)
    macs, params = profile(model, inputs=(input,), verbose=False)
    print('{:<40}  {:<8}'.format('Computational complexity (MACs):', macs))
    print('{:<40}  {:<8}'.format('Number of parameters:', params))


def count_ptflops(model, inputs_dim, tag):
    macs, params = get_model_complexity_info(model, inputs_dim, as_strings=False, print_per_layer_stat=False, )
    print('{:<10} {:<40}  {:<8}'.format(tag.upper(), 'Computational complexity (MACs):', macs))
    print('{:<10} {:<40}  {:<8}'.format(tag.upper(), 'Number of parameters:', params))
    return macs, params


def count_thop(model, inputs_dim, tag):
    input = torch.randn((1,) + inputs_dim)
    macs, params = profile(model, inputs=(input,), verbose=False)
    print('{:<10} {:<40}  {:<8}'.format(tag.upper(), 'Computational complexity (MACs):', macs))
    print('{:<10} {:<40}  {:<8}'.format(tag.upper(), 'Number of parameters:', params))
    return macs, params


class CPAH(nn.Module):
    def __init__(self, image_dim, text_dim, hidden_dim, hash_dim, label_dim):
        super(CPAH, self).__init__()
        self.module_name = 'CPAH'
        self.image_dim = image_dim
        self.text_dim = text_dim
        self.hidden_dim = hidden_dim
        self.hash_dim = hash_dim
        self.label_dim = label_dim

        class Unsqueezer(nn.Module):
            """
            Converts 2d input into 4d input for Conv2d layers
            """
            def __init__(self):
                super(Unsqueezer, self).__init__()

            def forward(self, x):
                return x.unsqueeze(1).unsqueeze(-1)

        self.mask_module = nn.ModuleDict({
            'img': nn.Sequential(
                Unsqueezer(),
                nn.Conv1d(1, hidden_dim, kernel_size=(hidden_dim, 1), stride=(1, 1)),
                nn.Sigmoid()
            ),
            'txt': nn.Sequential(
                Unsqueezer(),
                nn.Conv1d(1, hidden_dim, kernel_size=(hidden_dim, 1), stride=(1, 1)),
                nn.Sigmoid()
            ),
        })

        # D (consistency adversarial loss)
        self.feature_dis = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 8, bias=True),
            nn.ReLU(True),
            nn.Linear(hidden_dim // 8, 1, bias=True)
        )

        # C (consistency classification)
        self.consistency_dis = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 8, bias=True),
            nn.ReLU(True),
            nn.Linear(hidden_dim // 8, 3, bias=True)
        )

        # classification
        self.classifier = nn.ModuleDict({
            'img': nn.Sequential(
                nn.Linear(hidden_dim, label_dim, bias=True),
                nn.Sigmoid()
            ),
            'txt': nn.Sequential(
                nn.Linear(hidden_dim, label_dim, bias=True),
                nn.Sigmoid()
            ),
        })


class GEN(torch.nn.Module):
    def __init__(self, image_dim, hidden_dim, hash_dim):
        super(GEN, self).__init__()
        self.module_name = 'GEN_module'
        self.hash_dim = hash_dim

        class Unsqueezer(nn.Module):
            """
            Converts 2d input into 4d input for Conv2d layers
            """
            def __init__(self):
                super(Unsqueezer, self).__init__()

            def forward(self, x):
                return x.unsqueeze(1).unsqueeze(-1)

        self.main_module = nn.Sequential(
            nn.Linear(image_dim, hidden_dim, bias=True),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(True),
            nn.Linear(hidden_dim, hidden_dim, bias=True),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(True),
        )

        self.hash_module = nn.Sequential(
            nn.Linear(512, hash_dim, bias=True),
            nn.Tanh())

        self.mask_module = nn.Sequential(
            Unsqueezer(),
            nn.Conv1d(1, hidden_dim, kernel_size=(hidden_dim, 1), stride=(1, 1)),
            nn.Sigmoid()
        )

    def forward(self, x):
        f = self.main_module(x)  # image feature

        # MASKING
        mc = self.mask_module(f)

        f_ = f * mc

        # HASHING

        h_code = self.hash_module(f_)
        return h_code


class DIS(torch.nn.Module):
    def __init__(self, hidden_dim, hash_dim, out_dim):
        super(DIS, self).__init__()
        self.module_name = 'GEN_module'
        self.hash_dim = hash_dim

        self.dis = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 8, bias=True),
            nn.ReLU(True),
            nn.Linear(hidden_dim // 8, out_dim, bias=True)
        )

    def forward(self, x):
        f_x = self.dis(x)
        return f_x


class CLS(torch.nn.Module):
    def __init__(self, hidden_dim, label_dim):
        super(CLS, self).__init__()
        self.module_name = 'GEN_module'

        self.cls = nn.Sequential(
                nn.Linear(hidden_dim, label_dim, bias=True),
                nn.Sigmoid()
            )

    def forward(self, x):
        f_x = self.cls(x)
        return f_x


gi = GEN(512, 512, 128)
gt = GEN(768, 512, 128)
dd = DIS(512, 128, 1)
dc = DIS(512, 128, 3)
c = CLS(512, 31)

device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")

input_dims = (512,)
model = gi


def calculate_stats_for_unhd(method='ptflops'):
    if method == 'ptflops':

        f = count_ptflops
    else:
        f = count_thop

    print('\n\n\n' + method + '\n')
    print('Module stats:')
    mgi, pgi = f(gi, (512,), 'img')
    mgt, pgt = f(gt, (768,), 'txt')
    mdd, pdd = f(dd, (512,), 'dis d')
    mdc, pdc = f(dc, (512,), 'dis c')
    mc, pc = f(c, (512,), 'cls')

    total_params = pgi + pgt + pdd + pdc * 2 + pc * 2
    total_macs = mgi + mgt + mdd * 2 + mdc * 2 + mc * 2

    print('\nTotal stats:')
    print('{:<40}  {:<8}'.format('Computational complexity (MACs):', total_macs))
    print('{:<40}  {:<8}'.format('Computational complexity (FLOPs):', total_macs * 2))
    print('{:<40}  {:<8}'.format('Number of parameters:', total_params))


def calculate_stats():
    calculate_stats_for_unhd()
    calculate_stats_for_unhd('thop')


if device.type == 'cpu':
    # test_ptflops()
    # test_thop()
    calculate_stats()
else:
    with torch.cuda.device(device):
        # test_ptflops()
        # test_thop()
        calculate_stats()
