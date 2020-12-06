import torch.nn as nn
import torch
from cifar10_dataset import validation_loader


def validate_net(net: nn.Module) -> (float, float):
    correct = 0
    loss = 0.0

    with torch.no_grad():
        for data in validation_loader:
            images, labels = data
            images, labels = images.cuda(), labels.cuda()

            outputs = net(images)

            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum().item()

    total = len(validation_loader.dataset)
    return correct / total, loss / total
