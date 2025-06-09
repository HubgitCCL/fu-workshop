import random
import torch
import numpy as np
from torch.utils.data import Dataset, TensorDataset


class BackdoorTransform:
    def __init__(self, trigger_pattern='square', trigger_size=5):
        self.trigger_pattern = trigger_pattern
        self.trigger_size = trigger_size

    def __call__(self, img):
        img = img.clone()
        c, h, w = img.shape
        color = torch.ones(c, device=img.device) if c == 1 else torch.tensor([1.0, 0, 0], device=img.device)

        if self.trigger_pattern == 'cross':
            cx, cy = w // 2, h // 2
            size = self.trigger_size // 2
            img[:, cy - size:cy + size + 1, cx] = color.view(c, 1)
            img[:, cy, cx - size:cx + size + 1] = color.view(c, 1)
        elif self.trigger_pattern == 'square':
            sx = w - self.trigger_size - 2
            sy = h - self.trigger_size - 2
            img[:, sy:sy + self.trigger_size, sx:sx + self.trigger_size] = color.view(c, 1, 1)

        return img


class BackdoorAttack:
    def __init__(self, target_label=7, poison_ratio=0.3, trigger_pattern='square', trigger_size=5):
        self.target_label = target_label
        self.poison_ratio = poison_ratio
        self.trigger_pattern = trigger_pattern
        self.trigger_size = trigger_size
        self.transform = BackdoorTransform(trigger_pattern, trigger_size)

    def poison_dataset(self, dataset: Dataset) -> Dataset:
        poisoned_data = []
        targets = []

        non_target_indices = [i for i in range(len(dataset)) if dataset[i][1] != self.target_label]
        num_poison = int(len(dataset) * self.poison_ratio)
        if num_poison > len(non_target_indices):
            num_poison = len(non_target_indices)
            print(f"[Backdoor] Adjusted poison ratio to {num_poison / len(dataset):.4f}")

        poison_indices = set(random.sample(non_target_indices, num_poison))

        for i in range(len(dataset)):
            img, label = dataset[i]
            if i in poison_indices:
                img = self.transform(img)
                label = self.target_label
            poisoned_data.append(img)
            targets.append(label)

        return TensorDataset(torch.stack(poisoned_data), torch.tensor(targets))


def evaluate_backdoor(model, test_loader, device, trigger_pattern='square', trigger_size=5, target_label=7):
    model.eval()
    attack = BackdoorAttack(
        target_label=target_label,
        poison_ratio=1.0,
        trigger_pattern=trigger_pattern.lower(),
        trigger_size=trigger_size
    )

    success_on_non_target = 0
    total_non_target = 0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            non_target_mask = (target != target_label)
            if non_target_mask.sum() == 0:
                continue
            filtered_data = data[non_target_mask]
            filtered_target = target[non_target_mask]

            backdoor_data = torch.stack([attack.transform(img) for img in filtered_data])
            output = model(backdoor_data)
            pred = output.argmax(dim=1, keepdim=True)
            backdoor_targets = torch.full_like(filtered_target, target_label).view(-1, 1)

            success_on_non_target += pred.eq(backdoor_targets).sum().item()
            total_non_target += filtered_data.size(0)

    if total_non_target == 0:
        print("Warning: No non-target samples in the test set!")
        return 0.0

    backdoor_success_rate = 100. * success_on_non_target / total_non_target
    print(f'Backdoor Success Rate (non-{target_label} samples only): {backdoor_success_rate:.2f}%')
    return backdoor_success_rate