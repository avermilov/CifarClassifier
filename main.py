from torchvision.models import resnet18

import train
import validation
import test
import torch
from train import PATH
from nets import dropout_net, my_standard_net, my_net_leaky_relu, tutorial_net, my_net_tinkered_with
import torch.optim as optim
import torch.nn as nn
from time import time
from cifar10_dataset import train_loader

if __name__ == "__main__":
    # play around with different optimizers, loss functions, nets, etc

    # example:

    # initialize net, loss, optimizer
    net = my_standard_net.Net().cuda()
    # net = resnet18(num_classes=10).cuda()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.0005, momentum=0.9)

    # train
    start = time()
    train.train(net, 5, criterion, optimizer, log_to_tensorboard=True, train_name="simple_train_more_log",
                validation_name="simple_valid_more_log")

    # print validation and test accuracy
    print(f"Training time: {time() - start:.2f} seconds")
    print(f"Accuracy on validation: {validation.validate_net(net)[0] * 100:.2f}")
    test.test_net(net)
