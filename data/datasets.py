import torch
from torchvision import datasets, transforms

def load_data(data_dir, dataset_name):
    """加载数据集"""
    if dataset_name == 'mnist':
        return load_mnist(data_dir)
    elif dataset_name == 'cifar10':
        return load_cifar10(data_dir)
    else:
        raise ValueError('Invalid dataset name')
def load_mnist(data_dir):
    """加载MNIST数据集"""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_set = datasets.MNIST(
        root=data_dir, train=True, download=True, transform=transform)

    test_set = datasets.MNIST(
        root=data_dir, train=False, download=True, transform=transform)

    return train_set, test_set


def load_cifar10(data_dir):
    """加载CIFAR10数据集（扩展支持）"""
    # 类似MNIST的实现...
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    train_set = datasets.CIFAR10(
        root=data_dir, train=True, download=True, transform=transform)
    test_set = datasets.CIFAR10(
        root=data_dir, train=False, download=True, transform=transform)
    return train_set, test_set
