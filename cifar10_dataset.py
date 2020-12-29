import torch
import torchvision
import torchvision.transforms as transforms
from dataset_wrapper import expand_classification_dataset

# Change to True if first time running
CIFAR10_DOWNLOAD_DATASET = False
CIFAR10_BATCH_SIZE = 50
CIFAR10_VALIDATION_SIZE = 7000
CIFAR10_NUM_WORKERS = 8

cifar10_classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

cifar10_transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

cifar10_train_set = expand_classification_dataset(
    torchvision.datasets.CIFAR10(root='./data', train=True,
                                 download=CIFAR10_DOWNLOAD_DATASET, transform=cifar10_transform))

cifar10_train_loader = torch.utils.data.DataLoader(cifar10_train_set, batch_size=CIFAR10_BATCH_SIZE,
                                                   shuffle=True, num_workers=CIFAR10_NUM_WORKERS)

cifar10_valid_and_test_set = torchvision.datasets.CIFAR10(root='./data', train=False,
                                                          download=CIFAR10_DOWNLOAD_DATASET, transform=cifar10_transform)
cifar10_validation_loader = torch.utils.data.DataLoader([cifar10_valid_and_test_set[i] for i in range(CIFAR10_VALIDATION_SIZE)],
                                                        batch_size=CIFAR10_BATCH_SIZE,
                                                        shuffle=False, num_workers=CIFAR10_NUM_WORKERS)
cifar10_test_loader = torch.utils.data.DataLoader(
    [cifar10_valid_and_test_set[i] for i in range(CIFAR10_VALIDATION_SIZE, len(cifar10_valid_and_test_set))],
    batch_size=CIFAR10_BATCH_SIZE,
    shuffle=False, num_workers=2)
