"""dataset.py"""

import os
import random
import numpy as np

from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import ImageFolder, MNIST
from torchvision import transforms


def is_power_of_2(num):
    return ((num & (num - 1)) == 0) and num != 0


class CustomImageFolder(ImageFolder):
    def __init__(self, root, transform=None):
        super(CustomImageFolder, self).__init__(root, transform)
        self.indices = range(len(self))

    def __getitem__(self, index1):
        index2 = random.choice(self.indices)

        path1 = self.imgs[index1][0]
        path2 = self.imgs[index2][0]
        img1 = self.loader(path1)
        img2 = self.loader(path2)
        if self.transform is not None:
            img1 = self.transform(img1)
            img2 = self.transform(img2)

        return img1, img2


class CustomTensorDataset(Dataset):
    def __init__(self, data_tensor, transform=None):
        self.data_tensor = data_tensor
        self.transform = transform
        self.indices = range(len(self))

    def __getitem__(self, index1):
        index2 = random.choice(self.indices)

        img1 = self.data_tensor[index1]
        img2 = self.data_tensor[index2]
        if self.transform is not None:
            img1 = self.transform(img1)
            img2 = self.transform(img2)

        return img1, img2

    def __len__(self):
        return self.data_tensor.size(0)

class CustomMNISTDataset(MNIST):
    def __init__(self, *args, **kwargs):
        try:
            super().__init__(*args, **kwargs)
        except RuntimeError:
            super().__init__(*args, **kwargs, download=True)
        self.indices = range(len(self))
        
    def __getitem__(self, index1):
        index2 = random.choice(self.indices)
        if self.train:
            img1, img2 = self.train_data[index1], self.train_data[index2]
        else:
            img1, img2 = self.test_data[index1], self.test_data[index2]
        
        # c.f. from original MNIST fn, made to be consistent with other datasets
        img1 = Image.fromarray(img1.numpy(), mode='L')
        img2 = Image.fromarray(img2.numpy(), mode='L')

        if self.transform is not None:
            img1, img2 = self.transform(img1), self.transform(img2)

        return img1, img2

class ColoredMNISTDataset(MNIST):
    def __init__(
        self,
        class_prob=0.75,
        class_split=4,
        color_prob_train=0.9,
        color_prob_test=0.1,
        *args,
        **kwargs,
    ):
        try:
            super().__init__(*args, **kwargs)
        except RuntimeError:
            super().__init__(*args, **kwargs, download=True)
        self.indices = range(len(self))

        self.class_prob = 0.75
        self.class_split = class_split
        self.color_prob = color_prob_train if self.train else color_prob_test
        self.zeros = np.zeros_like(self.train_data[0])
        
    def __getitem__(self, index1):
        index2 = random.choice(self.indices)

        if self.train:
            img1, img2 = self.train_data[index1], self.train_data[index2]
            lab1, lab2 = self.train_labels[index1], self.train_labels[index2]
        else:
            img1, img2 = self.test_data[index1], self.test_data[index2]
            lab1, lab2 = self.test_labels[index1], self.test_labels[index2]

        # determine how to color the data
        # usually numbers zero to `class_split` belong in class 0
        # and numbers `class_split` + 1 to nine belong in class 1
        # we flip those classes with probability 1 - `class_prob`
        # (using the index so that these assignments are consistent)
        np.random.seed(index1)
        col1 = (int(lab1) > self.class_split) ^ (np.random.uniform() > self.class_prob)
        np.random.seed(index2)
        col2 = (int(lab2) > self.class_split) ^ (np.random.uniform() > self.class_prob)

        # make a 3-channel image
        # if class 0, channel 0 is colored, and same for class 1
        img1 = np.stack([img1 * (col1 == 0), img1 * (col1 == 1), self.zeros], axis=-1)
        img2 = np.stack([img2 * (col2 == 0), img2 * (col2 == 1), self.zeros], axis=-1)

        img1 = Image.fromarray(img1, 'RGB')
        img2 = Image.fromarray(img2, 'RGB')

        if self.transform is not None:
            img1, img2 = self.transform(img1), self.transform(img2)

        return img1, img2


def return_data(args):
    name = args.dataset
    dset_dir = args.dset_dir
    batch_size = args.batch_size
    num_workers = args.num_workers
    image_size = args.image_size
    assert image_size == 64, 'currently only image size of 64 is supported'

    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),])

    if name.lower() == 'celeba':
        root = os.path.join(dset_dir, 'CelebA')
        train_kwargs = {'root':root, 'transform':transform}
        dset = CustomImageFolder
    elif name.lower() == '3dchairs':
        root = os.path.join(dset_dir, '3DChairs')
        train_kwargs = {'root':root, 'transform':transform}
        dset = CustomImageFolder
    elif 'dsprites' in name.lower():
        if name.lower() == 'dsprites':
            root = os.path.join(dset_dir, 'dsprites-dataset/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz')
        if name.lower() == 'dsprites-colored':
            root = os.path.join(dset_dir, 'dsprites-dataset/dsprites_ndarray_colored.npz')
        data = np.load(root, encoding='latin1')
        data = torch.from_numpy(data['imgs']).unsqueeze(1).float()
        train_kwargs = {'data_tensor':data}
        dset = CustomTensorDataset
    elif name.lower() == 'mnist':
        root = os.path.join(dset_dir, 'mnist')
        train_kwargs = {'root': root, 'transform': transform}
        dset = CustomMNISTDataset        
    elif name.lower() == 'mnist-colored':
        root = os.path.join(dset_dir, 'mnist')
        train_kwargs = {'root': root, 'transform': transform}
        dset = ColoredMNISTDataset    

    else:
        raise NotImplementedError


    train_data = dset(**train_kwargs)
    train_loader = DataLoader(train_data,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=num_workers,
                              pin_memory=True,
                              drop_last=True)

    data_loader = train_loader
    return data_loader
