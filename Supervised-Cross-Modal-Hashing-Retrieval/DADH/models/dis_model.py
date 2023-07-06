import torch
from torch import nn
import torch.nn.init as init


class DIS(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, hash_dim):
        super(DIS, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.hash_dim = hash_dim

        self.feature_dis = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim, bias=True),
            nn.ReLU(True),
            nn.Linear(self.hidden_dim, self.hidden_dim // 2, bias=True),
            nn.ReLU(True),
            nn.Linear(self.hidden_dim//2, 1, bias=True)
        )

        self.hash_dis = nn.Sequential(
            nn.Linear(self.hash_dim, 512, bias=True),
            nn.ReLU(True),
            nn.Linear(512, 1, bias=True)
        )

        self.weight_init()

    def weight_init(self):
        initializer = self.kaiming_init
        for block in self._modules:
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

    def dis_feature(self, f):
        feature_score = self.feature_dis(f)
        return feature_score.squeeze()

    def dis_hash(self, h):
        hash_score = self.hash_dis(h)
        return hash_score.squeeze()
