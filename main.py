from torchvision.models import resnet18

import train
import validation
import test
import torch
from train import PATH
from nets import dropout_net, my_standard_net, my_net_leaky_relu, tutorial_net, my_net_tinkered_with, test_bigger_net, \
    even_bigger_net, experiment_net
import torch.optim as optim
import torch.nn as nn
from time import time
from cifar10_dataset import train_loader
from sys import exit

if __name__ == "__main__":
    # play around with different optimizers, loss functions, nets, etc

    # example:

    # initialize net, loss, optimizer
    # net = my_standard_net.Net().cuda()
    net = experiment_net.Net().cuda()
    # net = resnet18(num_classes=10).cuda()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.00015)

    # train
    start = time()
    train.train(net, 12, criterion, optimizer, log_to_tensorboard=True, train_name="res_net_train",
                validation_name="res_net_validation", save_name="best_res_net_model")

    # print validation and test accuracy
    print(f"Training time: {time() - start:.2f} seconds")
    print(f"Accuracy on validation: {validation.validate_net(net)[0] * 100:.2f}")
    test.test_net(net)
