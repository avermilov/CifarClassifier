import torch.nn as nn
import torch
from torch.utils.data import DataLoader
from device import DEVICE


def validate_net(net: nn.Module, criterion: nn.Module, validation_loader: DataLoader) -> (float, float):
    correct = 0
    loss = 0.0

    with torch.no_grad():
        for data in validation_loader:
            images, labels = data
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            outputs = net(images)
            loss += criterion(outputs, labels).item()

            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum().item()

    total = len(validation_loader.dataset)
    return correct / total, loss / total
