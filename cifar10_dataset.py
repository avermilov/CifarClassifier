import torch
import torchvision
import torchvision.transforms as transforms
from dataset_wrapper import expand_classification_dataset

# Change to True if first time running
DOWNLOAD_DATASET = False
BATCH_SIZE = 50
VALIDATION_SIZE = 7000
NUM_WORKERS = 8

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

train_set = expand_classification_dataset(
    torchvision.datasets.CIFAR10(root='./data', train=True,
                                 download=DOWNLOAD_DATASET, transform=transform))

train_loader = torch.utils.data.DataLoader(train_set, batch_size=BATCH_SIZE,
                                           shuffle=True, num_workers=NUM_WORKERS)

valid_and_test_set = torchvision.datasets.CIFAR10(root='./data', train=False,
                                                  download=DOWNLOAD_DATASET, transform=transform)
validation_loader = torch.utils.data.DataLoader([valid_and_test_set[i] for i in range(VALIDATION_SIZE)],
                                                batch_size=BATCH_SIZE,
                                                shuffle=False, num_workers=NUM_WORKERS)
test_loader = torch.utils.data.DataLoader(
    [valid_and_test_set[i] for i in range(VALIDATION_SIZE, len(valid_and_test_set))],
    batch_size=BATCH_SIZE,
    shuffle=False, num_workers=2)
