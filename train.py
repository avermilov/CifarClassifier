import torch
import torch.nn as nn
import torch.optim as optim
from cifar10_dataset import train_loader
from math import exp
from torch.utils.tensorboard import SummaryWriter

EVERY_N_MINI_BATCHES = 1250
PATH = './cifar_net_best_accuracy.pth'


def train(net: nn.Module, epochs: int, criterion: nn.Module, optimizer: optim.Optimizer,
          write_to_tensorboard=False) -> None:
    best_loss = None
    writer = SummaryWriter()
    for epoch in range(epochs):  # loop over the dataset multiple times
        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            inputs, labels = inputs.cuda(), labels.cuda()

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs).cuda()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % EVERY_N_MINI_BATCHES == EVERY_N_MINI_BATCHES - 1:
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / EVERY_N_MINI_BATCHES))
                # save best loss
                if best_loss is None or running_loss < best_loss:
                    best_loss = running_loss
                    torch.save(net.state_dict(), PATH.format())
                # log to tensorboard
                if write_to_tensorboard:
                    writer.add_scalar("mini-batch no./loss", running_loss / EVERY_N_MINI_BATCHES,
                                      epoch * 50 + (i + 1) // EVERY_N_MINI_BATCHES)
                running_loss = 0.0
    print('Finished Training')
    writer.close()
