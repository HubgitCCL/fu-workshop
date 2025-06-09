import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

# === MODEL CONFIGURATION ===
DATASET = 'cifar10'         # 'mnist' | 'fashion_mnist' | 'cifar10'
MODEL_NAME = 'resnet18'     # 'net_mnist' | 'resnet18' | 'resnet34' | 'resnet50' | 'resnet101' | 'resnet152' | 'vgg18'

# === MODEL SELECTION INTERFACE ===
def model_init(config):
    dataset = config["dataset"]
    model_name = config["model_name"]

    if dataset in ['mnist', 'fashion_mnist']:
        if model_name == 'net_mnist':
            return Net_mnist()
        elif model_name.startswith('resnet'):
            return load_resnet(model_name, input_channels=1)
        elif model_name == 'vgg18':
            return load_vgg18(input_channels=1)
        else:
            raise ValueError(f"Unsupported model '{model_name}' for dataset '{dataset}'")

    elif dataset == 'cifar10':
        if model_name == 'vgg18':
            return load_vgg18(input_channels=3)
        elif model_name.startswith('resnet'):
            return load_resnet(model_name, input_channels=3)
        else:
            raise ValueError(f"Unsupported model '{model_name}' for dataset '{dataset}'")

    else:
        raise ValueError(f"Unsupported dataset '{dataset}'")


# === NET_MNIST ===
class Net_mnist(nn.Module):
    def __init__(self):
        super(Net_mnist, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4*4*50, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4*4*50)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


# === LOAD RESNET (TORCHVISION) ===
def load_resnet(name, input_channels=3):
    num_classes = 10
    if name == 'resnet18':
        model = models.resnet18(weights=None, num_classes=num_classes)
    elif name == 'resnet34':
        model = models.resnet34(weights=None, num_classes=num_classes)
    elif name == 'resnet50':
        model = models.resnet50(weights=None, num_classes=num_classes)
    elif name == 'resnet101':
        model = models.resnet101(weights=None, num_classes=num_classes)
    elif name == 'resnet152':
        model = models.resnet152(weights=None, num_classes=num_classes)
    else:
        raise ValueError(f"Unsupported ResNet variant: {name}")

    # Modify conv1 and maxpool for all cases to make it CIFAR/MNIST-friendly
    model.conv1 = nn.Conv2d(input_channels, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()

    return model


# === LOAD VGG18 ===
def load_vgg18(input_channels=3):
    model = models.vgg18(weights=None)
    if input_channels != 3:
        features = list(model.features)
        features[0] = nn.Conv2d(input_channels, 64, kernel_size=3, padding=1)
        model.features = nn.Sequential(*features)
    model.classifier[6] = nn.Linear(4096, 10)
    return model
