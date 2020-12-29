import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from device import DEVICE
import itertools
import numpy as np
import matplotlib.pyplot as plt


def get_conf_mat_plot(conf_mat: torch.Tensor, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    conf_mat = conf_mat.float() / conf_mat.sum(axis=1)[:, np.newaxis]
    fig = plt.figure(figsize=(len(classes), len(classes)))
    plt.imshow(conf_mat, interpolation='nearest', cmap=cmap, figure=fig)
    plt.title(title, figure=fig)
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45, figure=fig)
    plt.yticks(tick_marks, classes, figure=fig)

    fmt = '.2f' if normalize else 'd'
    thresh = conf_mat.max() / 2.
    for i, j in itertools.product(range(conf_mat.shape[0]), range(conf_mat.shape[1])):
        plt.text(j, i, format(conf_mat[i, j], fmt), horizontalalignment="center",
                 color="white" if conf_mat[i, j] > thresh else "black", figure=fig)

    plt.tight_layout()
    plt.ylabel('True label', figure=fig)
    plt.xlabel('Predicted label', figure=fig)
    return fig


@torch.no_grad()
def get_preds_and_labels(net: nn.Module, data_loader: DataLoader) -> (torch.Tensor, torch.Tensor):
    all_preds = torch.Tensor([]).to(DEVICE)
    all_labels = torch.Tensor([]).to(DEVICE)
    for batch in data_loader:
        images, labels = batch
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        preds = net(images)
        all_preds = torch.cat((all_preds, preds), dim=0)
        all_labels = torch.cat((all_labels, labels), dim=0)
    return all_preds, all_labels


def get_conf_mat(net: nn.Module, data_loader: DataLoader, num_classes: int) -> torch.Tensor:
    train_preds, train_labels = get_preds_and_labels(net, data_loader)
    mat = torch.zeros(num_classes, num_classes, dtype=torch.int64)
    for tl, pl in zip(train_labels, train_preds.argmax(dim=1)):
        mat[int(tl), int(pl)] += 1
    return mat
