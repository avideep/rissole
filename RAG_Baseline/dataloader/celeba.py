import os
import random
import torch
import torchvision
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.data import SubsetRandomSampler
import numpy as np
from PIL import Image
import torch.nn.functional as F
from tqdm import tqdm

# gpus = tf.config.list_physical_devices('GPU')
# tf.config.set_visible_devices(gpus[0], 'GPU')
class CelebA:
    def __init__(self, root = '/hdd/avideep/blockLDM/data/', batch_size: int = 16, dset_batch_size: int = 32, img_size = 64, block_size = 16):
        """
        Wrapper to load, preprocess and deprocess CIFAR-10 dataset.
        Args:
            batch_size (int): Batch size, default: 16.
        """
        # self.classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
        # self.n_classes = 10

        self.batch_size = batch_size
        self.img_size = img_size
        self.mean = [0.5, 0.5, 0.5]
        self.std = [0.5, 0.5, 0.5]
        self.train_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(img_size),
            transforms.CenterCrop(img_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.Normalize(self.mean, self.std)
        ])
        self.val_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(64),
            transforms.CenterCrop(64),
            transforms.Normalize(self.mean, self.std)
        ])
        self.ROOT_PATH = root
        self.IMAGE_PATH = root + 'celeba/processed/'
        train_size = int(50000 * 0.9)
        # val size is 10000 in cifar10
        test_size = 50000 - train_size

        self.train_set_full = ImageFolder(self.IMAGE_PATH, self.train_transform)
        self.train_loader, self.val_loader = self.train_val_test_split(self.train_set_full, self.batch_size)
        self.full_dataloader = DataLoader(self.train_set_full, batch_size=dset_batch_size, num_workers=4, pin_memory=True)
        # invert normalization for tensor to image transform
        self.inv_normalize = transforms.Compose([
            transforms.Normalize(mean=0, std=[1./s for s in self.std]),
            transforms.Normalize(mean=[-m for m in self.mean], std=1.),
            lambda x: x*255
        ])
        
    def train_val_test_split(self, dataset, batch_size=16, test_ratio=0.2, val_ratio=0.2):

        np.random.seed(42)

        data_indices = np.arange(len(dataset))
        #print('data_indices = ', data_indices)
        np.random.shuffle(data_indices)

        val_size = int(np.floor(len(data_indices) * val_ratio))
        #print('test_size = ', test_size )
        train_size = len(data_indices) - val_size

        all_indices = list(data_indices)
        val_indices = random.sample(all_indices, val_size)
        
        train_indices = np.setdiff1d(list(all_indices), val_indices)
        
        train_indices = list(train_indices)

        train_loader = DataLoader(dataset, batch_size=batch_size, sampler=SubsetRandomSampler(train_indices), num_workers=4, pin_memory=True)

        val_loader = DataLoader(dataset, batch_size=batch_size, sampler=SubsetRandomSampler(val_indices), num_workers=4, pin_memory=True)

        # test_loader = DataLoader(dataset, batch_size=batch_size, sampler=SubsetRandomSampler(test_indices), num_workers=12, pin_memory=True)

        return train_loader, val_loader

    @property
    def train(self):
        """ Return training dataloader. """
        return self.train_loader

    @property
    def val(self):
        """ Return validation dataloader. """
        return self.val_loader

    # @property
    # def test(self):
    #     """ Return test dataloader. """
    #     return self.test_loader

    def idx2label(self, idx):
        """ Return class label for given index. """
        return self.classes[idx]

    def prob2label(self, prob_vector):
        """ Return class label with highest confidence. """
        return self.idx2label(torch.argmax(prob_vector).cpu())

    def tensor2img(self, tensor):
        """ Convert torch.Tensor to PIL image. """
        n_channels = tensor.shape[0]

        img = tensor.detach().cpu()
        img = self.inv_normalize(img)

        if n_channels > 1:
            return Image.fromarray(img.permute(1, 2, 0).numpy().astype('uint8')).convert("RGB")
        else:
            return Image.fromarray(img[0].numpy()).convert("L")
    
class CelebAHQ:
    def __init__(self, root = '/hdd/avideep/blockLDM/data/', batch_size: int = 16, dset_batch_size: int = 32, img_size = 256, block_size = 32, device = None):
        """
        Wrapper to load, preprocess and deprocess CIFAR-10 dataset.
        Args:
            batch_size (int): Batch size, default: 16.
        """
        # self.classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
        # self.n_classes = 10
        self.img_size = img_size
        self.batch_size = batch_size
        self.device = device
        self.mean = [0.5, 0.5, 0.5]
        self.std = [0.5, 0.5, 0.5]
        self.train_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(img_size),
            transforms.CenterCrop(img_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.Normalize(self.mean, self.std)
        ])

        self.val_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(256),
            transforms.CenterCrop(256),
            transforms.Normalize(self.mean, self.std)
        ])
        self.ROOT_PATH = root
        self.IMAGE_PATH = root + 'celeba/celeba_hq/images'
        train_size = int(50000 * 0.9)
        # val size is 10000 in cifar10
        test_size = 50000 - train_size

        self.train_set_full = ImageFolder(self.IMAGE_PATH, self.train_transform)
        self.train_loader, self.val_loader = self.train_val_test_split(self.train_set_full, self.batch_size)
        self.full_dataloader = DataLoader(self.train_set_full, batch_size=dset_batch_size, num_workers=12, pin_memory=True)
        # invert normalization for tensor to image transform
        self.inv_normalize = transforms.Compose([
            transforms.Normalize(mean=0, std=[1./s for s in self.std]),
            transforms.Normalize(mean=[-m for m in self.mean], std=1.),
            lambda x: x*255
        ])
    def train_val_test_split(self, dataset, batch_size=16, test_ratio=0.2, val_ratio=0.2):

        np.random.seed(42)

        data_indices = np.arange(len(dataset))
        #print('data_indices = ', data_indices)
        np.random.shuffle(data_indices)

        val_size = int(np.floor(len(data_indices) * val_ratio))
        #print('test_size = ', test_size )
        train_size = len(data_indices) - val_size

        all_indices = list(data_indices)
        val_indices = random.sample(all_indices, val_size)
        
        train_indices = np.setdiff1d(list(all_indices), val_indices)
        
        train_indices = list(train_indices)

        train_loader = DataLoader(dataset, batch_size=batch_size, sampler=SubsetRandomSampler(train_indices), num_workers=12, pin_memory=True)

        val_loader = DataLoader(dataset, batch_size=batch_size, sampler=SubsetRandomSampler(val_indices), num_workers=12, pin_memory=True)

        # test_loader = DataLoader(dataset, batch_size=batch_size, sampler=SubsetRandomSampler(test_indices), num_workers=12, pin_memory=True)

        return train_loader, val_loader

    @property
    def train(self):
        """ Return training dataloader. """
        return self.train_loader

    @property
    def val(self):
        """ Return validation dataloader. """
        return self.val_loader

    # @property
    # def test(self):
    #     """ Return test dataloader. """
    #     return self.test_loader

    def idx2label(self, idx):
        """ Return class label for given index. """
        return self.classes[idx]

    def prob2label(self, prob_vector):
        """ Return class label with highest confidence. """
        return self.idx2label(torch.argmax(prob_vector).cpu())

    def tensor2img(self, tensor):
        """ Convert torch.Tensor to PIL image. """
        n_channels = tensor.shape[0]

        img = tensor.detach().cpu()
        img = self.inv_normalize(img)

        if n_channels > 1:
            return Image.fromarray(img.permute(1, 2, 0).numpy().astype('uint8')).convert("RGB")
        else:
            return Image.fromarray(img[0].numpy()).convert("L")


if __name__ == "__main__":
    data = CelebA(batch_size=16)

    for batch in data.train:
        ims, labels = batch

        print("images")
        print("\t", ims.shape)
        print(f"\t {ims.min()} < {torch.mean(ims)} < {ims.max()}")
        print("labels")
        print("\t", labels)

        ims = ims.detach().cpu()
        ims_grid = torchvision.utils.make_grid(ims)

        pil_ims = data.tensor2img(ims_grid)
        SAVE_DIR = 'datasamples'
        if not os.path.exists(SAVE_DIR):
            os.mkdir(SAVE_DIR)
        pil_ims.save(os.path.join(SAVE_DIR, 'celeba_sample.png'))

        break
