import torch
import torchvision
import torchvision.transforms as transforms

# Change to True if first time running
DOWNLOAD_DATASET = False
BATCH_SIZE = 4
VALIDATION_SIZE = 7000

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

train_set = torchvision.datasets.CIFAR10(root='./data', train=True,
                                         download=DOWNLOAD_DATASET, transform=transform)
train_loader = torch.utils.data.DataLoader(train_set, batch_size=BATCH_SIZE,
                                           shuffle=True, num_workers=2)

valid_and_test_set = torchvision.datasets.CIFAR10(root='./data', train=False,
                                                  download=DOWNLOAD_DATASET, transform=transform)
validation_loader = torch.utils.data.DataLoader([valid_and_test_set[i] for i in range(VALIDATION_SIZE)],
                                                batch_size=BATCH_SIZE,
                                                shuffle=False, num_workers=2)

test_loader = torch.utils.data.DataLoader(
    [valid_and_test_set[i] for i in range(VALIDATION_SIZE, len(valid_and_test_set))],
    batch_size=BATCH_SIZE,
    shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
