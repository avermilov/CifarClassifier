import matplotlib.pyplot as plt
import numpy as np
import torchvision
from cifar10_dataset import cifar10_train_loader, cifar10_classes


def imshow(img):
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


if __name__ == "__main__":
    # get some random training images
    dataiter = iter(cifar10_train_loader)
    images, labels = dataiter.next()

    # show images
    imshow(torchvision.utils.make_grid(images))
    # print labels
    print(' '.join('%5s' % cifar10_classes[labels[j]] for j in range(4)))
