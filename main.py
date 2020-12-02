import train
import validation
import test
import torch
from train import PATH
from nets import dropout_net, my_standard_net, my_net_leaky_relu, tutorial_net, my_net_tinkered_with
import torch.optim as optim
import torch.nn as nn

if __name__ == "__main__":
    # play around with different optimizers, loss functions, nets, etc

    # example:

    # initialize net, loss, optimizer
    net = my_standard_net.Net().cuda()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    # train
    train.train(net, 1, criterion, optimizer, write_to_tensorboard=True)

    # print validation and test accuracy
    print(f"Accuracy on validation: {validation.validate_net(net) * 100:.2f}")
    test.test_net(net)
