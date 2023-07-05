from itertools import chain

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader
from torchvision.transforms import transforms

from datasets.Mirflickr25kDataset import Mirflickr25kDataset, split_dataset
from models.customalexnet import CustomAlexNet

torch.cuda.set_device(11)


def train(model, device, train_loader, optimizer, criterion, epoch):
    model.train()
    for batch_idx, (data, label, _) in enumerate(train_loader):
        data, label = data.to(device), label.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, label)
        loss.backward()
        optimizer.step()

        if batch_idx % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch, batch_idx * len(data), len(train_loader.dataset),
                                                                           100. * batch_idx / len(train_loader), loss.item()),
                  end=' === ')
            print('with learning rate:', optimizer.param_groups[0]['lr'])


@torch.no_grad()
def test(model, device, test_loader, criterion):
    model.eval()
    test_loss = 0
    correct = 0

    for data, label, _ in test_loader:
        data, label = data.to(device), label.to(device)
        output = model(data)
        test_loss += criterion(output, label).item()
        pred = output.argmax(1)

        un_one_hot_label = label.argmax(1)
        correct += pred.eq(un_one_hot_label.view_as(pred)).sum().item()

    test_loss /= len(test_loader)
    acc = 100. * correct / len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.4f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset), acc))


def finetune():
    batch_size = 1280
    epochs = 60
    device = torch.device('cuda')

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])

    dataset = Mirflickr25kDataset(transform=transform)
    train_set, test_set = split_dataset(dataset)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)

    model = CustomAlexNet(24)
    model.to(device)

    # optimizer = optim.SGD(
    #     [
    #         {'params': model.model.features.parameters(), 'lr': 0.001},
    #         {'params': chain(*[c.parameters() for c in model.model.classifier[:6]]), 'lr': 0.002},
    #         {'params': model.model.classifier[6].parameters(), 'lr': 0.01}
    #     ], momentum=0.9, weight_decay=0.0005, lr=0.002
    # )

    optimizer = optim.SGD(model.parameters(), momentum=0.9, weight_decay=0.0005, lr=0.01)

    criterion = nn.MSELoss()

    for epoch in range(epochs):
        train(model, device, train_loader, optimizer, criterion, epoch=epoch)
        test(model, device, test_loader, criterion)

    torch.std(model)


if __name__ == '__main__':
    finetune()
