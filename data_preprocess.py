import os
import numpy as np
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset, random_split, Subset
from typing import Tuple, List, Dict, Optional
from backdoor import BackdoorAttack

class Cutout(object):
    def __init__(self, n_holes: int, length: int):
        self.n_holes = n_holes
        self.length = length

    def __call__(self, img):
        h = img.size(1)
        w = img.size(2)

        mask = np.ones((h, w), np.float32)

        for n in range(self.n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)

            y1 = np.clip(y - self.length // 2, 0, h)
            y2 = np.clip(y + self.length // 2, 0, h)
            x1 = np.clip(x - self.length // 2, 0, w)
            x2 = np.clip(x + self.length // 2, 0, w)

            mask[y1:y2, x1:x2] = 0.

        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img = img * mask
        return img


def load_dataset(dataset_name: str, data_dir: str = './data') -> Tuple[Dataset, Dataset]:
    if dataset_name.lower() == 'mnist':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

        train_dataset = datasets.MNIST(
            root=data_dir,
            train=True,
            download=True,
            transform=transform
        )

        test_dataset = datasets.MNIST(
            root=data_dir,
            train=False,
            download=True,
            transform=transform
        )

    elif dataset_name.lower() == 'fashion_mnist':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.2860,), (0.3530,))
        ])

        train_dataset = datasets.FashionMNIST(
            root=data_dir,
            train=True,
            download=True,
            transform=transform
        )

        test_dataset = datasets.FashionMNIST(
            root=data_dir,
            train=False,
            download=True,
            transform=transform
        )

    elif dataset_name.lower() == 'cifar10':
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
            # Cutout(n_holes=1, length=16) # may damage the backdoor triggers
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
        ])

        train_dataset = datasets.CIFAR10(
            root=data_dir,
            train=True,
            download=True,
            transform=transform_train
        )

        test_dataset = datasets.CIFAR10(
            root=data_dir,
            train=False,
            download=True,
            transform=transform_test
        )

    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    return train_dataset, test_dataset


def create_iid_splits(dataset: Dataset, num_clients: int) -> List[Dataset]:
    total_size = len(dataset)
    partition_size = total_size // num_clients

    lengths = [partition_size] * (num_clients - 1)
    lengths.append(total_size - sum(lengths))

    partitions = random_split(dataset, lengths)

    return partitions


def create_non_iid_splits(dataset: Dataset, num_clients: int, 
                         shards_per_client: int = 2) -> List[Dataset]:
    if not hasattr(dataset, 'targets'):
        raise ValueError("Dataset does not have a 'targets' attribute")

    if isinstance(dataset.targets, torch.Tensor):
        labels = dataset.targets.numpy()
    else:
        labels = np.array(dataset.targets)

    indices = np.argsort(labels)

    num_classes = len(np.unique(labels))

    total_shards = num_classes

    shard_size = len(dataset) // total_shards
    shard_indices = [indices[i:i+shard_size] for i in range(0, len(indices), shard_size)]

    client_datasets = []
    shard_assignment = []

    for i in range(num_clients):
        client_shards = []
        while len(client_shards) < shards_per_client:
            shard_idx = np.random.randint(0, total_shards)
            if shard_idx not in shard_assignment:
                client_shards.append(shard_idx)
                shard_assignment.append(shard_idx)

                if len(shard_assignment) == total_shards:
                    shard_assignment = []

        client_indices = np.concatenate([shard_indices[s] for s in client_shards])

        client_datasets.append(Subset(dataset, client_indices))

    return client_datasets


def create_data_loaders(partitions: List[Dataset], 
                        batch_size: int = 32, 
                        test_dataset: Optional[Dataset] = None) -> Tuple[List[DataLoader], Optional[DataLoader]]:
    train_loaders = [
        DataLoader(partition, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
        for partition in partitions
    ]

    test_loader = None
    if test_dataset is not None:
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    return train_loaders, test_loader


def get_partitioned_data(dataset_name: str,
                        num_clients: int, 
                        batch_size: int = 32,
                        partition_type: str = 'iid',
                        shards_per_client: int = 2,
                        data_dir: str = './data',
                        backdoor_client: int = None,
                        backdoor_target_label: int = 7,
                        backdoor_poison_ratio: float = 0.3,
                        backdoor_trigger_pattern: str = 'cross',
                        backdoor_trigger_size: int = 5) -> Tuple[List[DataLoader], DataLoader]:
    train_dataset, test_dataset = load_dataset(dataset_name, data_dir)

    if partition_type == 'iid':
        partitions = create_iid_splits(train_dataset, num_clients)
    elif partition_type == 'non-iid':
        partitions = create_non_iid_splits(train_dataset, num_clients, shards_per_client)
    else:
        raise ValueError(f"Unknown partition type: {partition_type}")

    if backdoor_client is not None and 0 <= backdoor_client < num_clients:
        print(f"Implementing backdoor attack on client {backdoor_client}")
        print(f"Backdoor parameters: target_label={backdoor_target_label}, poison_ratio={backdoor_poison_ratio}, "
              f"trigger_pattern={backdoor_trigger_pattern}, trigger_size={backdoor_trigger_size}")

        attack = BackdoorAttack(
            target_label=backdoor_target_label,
            poison_ratio=backdoor_poison_ratio,
            trigger_pattern=backdoor_trigger_pattern,
            trigger_size=backdoor_trigger_size
        )

        partitions[backdoor_client] = attack.poison_dataset(partitions[backdoor_client])

    train_loaders, test_loader = create_data_loaders(partitions, batch_size, test_dataset)

    return train_loaders, test_loader
