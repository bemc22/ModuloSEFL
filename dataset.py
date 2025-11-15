import torch 
import numpy as np

from torchvision import transforms
from torchvision.datasets import DatasetFolder

def load_img(path):
    img = np.load(path)
    img = img.astype(np.float32)
    return img


class NormalizeRange(object):
    # Normalize a tensor image to a given range.
    def __init__(self, min_val=0.0, max_val=1.0):
        self.min_val = min_val
        self.max_val = max_val

    def __call__(self, tensor):

        tensor = tensor.float()
        # remove negative values
        tensor = torch.clamp(tensor, min=0.0)
        tensor = tensor - torch.min(tensor)
        tensor = tensor / torch.max(tensor)
        tensor = tensor * (self.max_val - self.min_val) + self.min_val
        return tensor


def load_dataset(root, max_val=4.0):
    

    train_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.RandomCrop(240),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        NormalizeRange(0.0, max_val),
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        NormalizeRange(0.0, max_val),
    ])


    train_path = root
    train_dataset = DatasetFolder(train_path, loader=load_img, extensions='.npy', transform=train_transform)

    test_path = root + '_test'
    test_dataset = DatasetFolder(test_path, loader=load_img, extensions='.npy', transform=test_transform)

    return train_dataset, test_dataset