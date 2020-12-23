import torch.nn as nn
import torch
from typing import List, Union
from torch.utils.data import DataLoader
from device import DEVICE


def test_net(net: nn.Module, test_loader: DataLoader, classes: Union[List, tuple]) -> (float, List[float]):
    correct = 0
    total = 0
    class_correct = list(0. for _ in range(len(classes)))
    class_total = list(0. for _ in range(len(classes)))

    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            outputs = net(images)
            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            c = (predicted == labels).squeeze()

            for i in range(test_loader.batch_size):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1
    return correct / total, {classes[i]: class_correct[i] / class_total[i] for i in range(len(classes))}
