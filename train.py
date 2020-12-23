import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from validation import validate_net
from device import DEVICE

EVERY_N_MINI_BATCHES = 125
PATH = 'best_results/'


def train(net: nn.Module, epochs: int, criterion: nn.Module, optimizer: optim.Optimizer, train_loader: DataLoader,
          log_to_tensorboard: bool = False, train_name: str = "train", validation_name: str = "validation",
          save_name: str = "cifar_model_best") -> None:
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

            # save best
            if i % EVERY_N_MINI_BATCHES == EVERY_N_MINI_BATCHES - 1:
                # save best loss
                validation_accuracy, validation_loss = validate_net(net, criterion=criterion)
                if best_validation_accuracy is None or validation_accuracy > best_validation_accuracy:
                    best_validation_accuracy = validation_accuracy
                    torch.save(net.state_dict(), PATH + save_name)

                if log_to_tensorboard:
                    writer.add_scalar(train_name + "/train_loss", running_loss / total, minibatch_no)
                    writer.add_scalar(train_name + "/train_accuracy", correct / total, minibatch_no)
                    writer.add_scalar(validation_name + "/validation_accuracy", validation_accuracy, minibatch_no)
                    writer.add_scalar(validation_name + "/validation_loss", validation_loss, minibatch_no)
                minibatch_no += 1

    writer.close()
