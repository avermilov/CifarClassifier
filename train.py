import torch
import torch.nn as nn
import torch.optim as optim
from cifar10_dataset import train_loader
from torch.utils.tensorboard import SummaryWriter
from validation import validate_net

EVERY_N_MINI_BATCHES = 1250
PATH = './cifar_net_best_accuracy.pth'


def train(net: nn.Module, epochs: int, criterion: nn.Module, optimizer: optim.Optimizer,
          log_to_tensorboard=False, train_name="train", validation_name="validation") -> None:
    best_validation_accuracy = None
    writer = SummaryWriter()
    total = len(train_loader.dataset)
    batch_no = 0
    for epoch in range(epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data
            inputs, labels = inputs.cuda(), labels.cuda()
            total += labels.size(0)

            optimizer.zero_grad()

            outputs = net(inputs).cuda()

            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum().item()

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            # save best
            if i % EVERY_N_MINI_BATCHES == EVERY_N_MINI_BATCHES - 1:
                # save best loss
                validation_accuracy, validation_loss = validate_net(net)
                if best_validation_accuracy is None or validation_accuracy < best_validation_accuracy:
                    best_validation_accuracy = validation_accuracy
                    torch.save(net.state_dict(), PATH.format())

                if log_to_tensorboard:
                    writer.add_scalar(train_name + "/train_loss", running_loss / total, batch_no)
                    writer.add_scalar(train_name + "/train_accuracy", correct / total, batch_no)
                    writer.add_scalar(validation_name + "/validation_accuracy", validation_accuracy, batch_no)
                    writer.add_scalar(validation_name + "/validation_loss", validation_loss, batch_no)
                batch_no += 1

    writer.close()
