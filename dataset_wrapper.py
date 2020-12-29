import torch
from collections import Counter


def expand_classification_dataset(dataset: torch.utils.data.Dataset, add_targets = False) -> torch.utils.data.Dataset:
    expanded_dataset = dataset
    expanded_dataset.occurences = dict(Counter(expanded_dataset.targets))
    expanded_dataset.cifar10_classes = expanded_dataset.occurences.keys()
    expanded_dataset.num_classes = len(expanded_dataset.cifar10_classes)
    return expanded_dataset
