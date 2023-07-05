import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.models import alexnet


class CustomAlexNet(torch.nn.Module):
    def __init__(self, num_classes=1000):
        super(CustomAlexNet, self).__init__()
        self.model = alexnet(pretrained=False)
        self.num_classes = num_classes
        self.model._modules['classifier'][6] = self._get_appended_layer()

    def _get_appended_layer(self):
        appended_layer = nn.Linear(
            self.model._modules['classifier'][4].out_features,
            self.num_classes
        )
        appended_layer.weight.data.normal_(0, 0.001)

        return appended_layer

    def forward(self, x):
        return F.softmax(self.model(x), dim=1)


if __name__ == '__main__':
    model = CustomAlexNet(24)
    print(model._modules)
    # for key, value in model.named_modules():
    #     print(key, value)
