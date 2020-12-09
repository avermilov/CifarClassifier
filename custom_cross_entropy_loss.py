import torch.nn as nn
import torch


class CustomCrossEntropyLoss(nn.Module):
    def __init__(self):
        super(CustomCrossEntropyLoss, self).__init__()

    def forward(self, labels: torch.tensor, truth: torch.tensor) -> torch.tensor:
        total_loss = 0.0
        for i in range(truth.shape[0]):
            print(-labels[i, truth[i]].item(), torch.log(torch.sum(torch.exp(labels[i]))).item())
            total_loss += -labels[i, truth[i]] + torch.log(torch.sum(torch.exp(labels[i])))
        return total_loss / truth.shape[0]