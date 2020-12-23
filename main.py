from torchvision.models import resnet18

import train
import validation
import test
import torch
from train import PATH
from nets import dropout_net, my_standard_net, my_net_leaky_relu, tutorial_net, my_net_tinkered_with, test_bigger_net, \
    even_bigger_net, densenet
import torch.optim as optim
import torch.nn as nn
from time import time
from cifar10_dataset import *
from sys import exit
import cm
from device import DEVICE
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
import numpy as np

if __name__ == "__main__":
    net = densenet.DenseNet121().to(DEVICE)
    net.load_state_dict(torch.load("best_results/best_densenet_big_batch_densenet"))
    print(test.test_net(net, test_loader, classes))
    exit()
    # play around with different optimizers, loss functions, nets, etc

    # example:

    net = densenet.DenseNet121().to(DEVICE)
    net.load_state_dict(torch.load("best_results/best_densenet_big_batch_densenet"))
    test.test_net(net)
    exit()

    # initialize net, loss, optimizer
    # net = my_standard_net.Net().to(DEVICE)
    net = densenet.DenseNet121().to(DEVICE)
    # net = resnet18(num_classes=10).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.00015)

    # train
    start = time()
    train.train(net, 20, criterion, optimizer, log_to_tensorboard=True, train_name="densenet_big_batch_train",
                validation_name="densenet_big_batch_validation", save_name="best_densenet_big_batch_densenet")

    # print validation and test accuracy
    print(f"Training time: {time() - start:.2f} seconds")
    print(f"Accuracy on validation: {validation.validate_net(net)[0] * 100:.2f}")
    test.test_net(net)
