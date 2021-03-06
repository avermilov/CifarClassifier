from typing import Union, List

import torch
import torch.nn as nn
import torch.optim as optim
import conf_mat
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from validation import validate_net
from device import DEVICE

CIFAR10_EVERY_N_MINI_BATCHES = 125
PATH = 'best_results/'


def train(net: nn.Module, epochs: int, criterion: nn.Module, optimizer: optim.Optimizer,
          train_loader: DataLoader, validation_loader: DataLoader, classes: Union[List, tuple],
          tb_log_graphs: bool = False, tb_log_conf_mat=False, train_folder_name: str = "train",
          validation_folder_name: str = "validation",
          model_save_filename: str = "model_best", confusion_matrix_name: str = "confusion_matrix") -> None:
    best_validation_accuracy = None
    writer = SummaryWriter()
    minibatch_no = 0
    for epoch in range(epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            total += labels.size(0)

            optimizer.zero_grad()

            outputs = net(inputs).to(DEVICE)

            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum().item()

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if i % CIFAR10_EVERY_N_MINI_BATCHES == CIFAR10_EVERY_N_MINI_BATCHES - 1:
                validation_accuracy, validation_loss = validate_net(net, criterion, validation_loader)
                if best_validation_accuracy is None or validation_accuracy > best_validation_accuracy:
                    best_validation_accuracy = validation_accuracy
                    torch.save(net.state_dict(), PATH + model_save_filename)
                if tb_log_conf_mat:
                    fig = conf_mat.get_conf_mat_plot(conf_mat.get_conf_mat(net, validation_loader, len(classes)),
                                                     classes, normalize=True)
                    writer.add_figure(confusion_matrix_name, fig, minibatch_no)
                if tb_log_graphs:
                    writer.add_scalar(train_folder_name + "/train_loss", running_loss / total, minibatch_no)
                    writer.add_scalar(train_folder_name + "/train_accuracy", correct / total, minibatch_no)
                    writer.add_scalar(validation_folder_name + "/validation_accuracy", validation_accuracy,
                                      minibatch_no)
                    writer.add_scalar(validation_folder_name + "/validation_loss", validation_loss, minibatch_no)
                minibatch_no += 1

    writer.close()
