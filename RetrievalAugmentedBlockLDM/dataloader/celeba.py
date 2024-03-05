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
from sentence_transformers import SentenceTransformer, util
from tqdm import tqdm
import scann
import tensorflow as tf

# gpus = tf.config.list_physical_devices('GPU')
# tf.config.set_visible_devices(gpus[0], 'GPU')
class CelebA:
    def __init__(self, batch_size: int = 16, img_size = 64, searcher_dir = None):
        """
        Wrapper to load, preprocess and deprocess CIFAR-10 dataset.
        Args:
            batch_size (int): Batch size, default: 16.
        """
        # self.classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
        # self.n_classes = 10

        self.batch_size = batch_size

        self.mean = [0.5, 0.5, 0.5]
        self.std = [0.5, 0.5, 0.5]
        self.patch_size = img_size // 2
        self.patches = 5
        self.train_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(img_size),
            transforms.CenterCrop(img_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.Normalize(self.mean, self.std)
        ])
        self.encoder = SentenceTransformer('clip-ViT-B-32')
        self.val_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(64),
            transforms.CenterCrop(64),
            transforms.Normalize(self.mean, self.std)
        ])
        self.ROOT_PATH = '/hdd/avideep/blockLDM/data/celeba/'
        self.IMAGE_PATH = '/hdd/avideep/blockLDM/data/celeba/processed/'
        self.DSET_PATH = '/hdd/avideep/blockLDM/data/celeba/dset.pth'
        train_size = int(50000 * 0.9)
        # val size is 10000 in cifar10
        test_size = 50000 - train_size

        self.train_set_full = ImageFolder(self.IMAGE_PATH, self.train_transform)
        self.train_loader, self.val_loader = self.train_val_test_split(self.train_set_full, self.batch_size)
        self.full_dataloader = DataLoader(self.train_set_full, batch_size=1, num_workers=12, pin_memory=True)
        # invert normalization for tensor to image transform
        self.inv_normalize = transforms.Compose([
            transforms.Normalize(mean=0, std=[1./s for s in self.std]),
            transforms.Normalize(mean=[-m for m in self.mean], std=1.),
            lambda x: x*255
        ])
        self.dset = self.dsetbuilder()
        searcher_dir = '/hdd/avideep/blockLDM/data/celeba/searcher/'
        if not os.path.exists(searcher_dir):
            self.searcher = scann.scann_ops_pybind.builder(self.dset / np.linalg.norm(self.dset, axis=1)[:, np.newaxis], 10, "dot_product").tree(num_leaves=2000, num_leaves_to_search=100, training_sample_size=250000).score_ah(2, anisotropic_quantization_threshold=0.2).reorder(100).build()
            print(f'Save trained searcher under "{searcher_dir}"')
            os.makedirs(searcher_dir, exist_ok=True)
            self.searcher.serialize(searcher_dir)
        else:
            print(f'Loading pre-trained searcher from {searcher_dir}')
            self.searcher = scann.scann_ops_pybind.load_searcher(searcher_dir).gpu()
            print('Finished loading searcher.')
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
    
    @property
    def dset(self):
        return self.dset

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
    def get_encodings(self, x):
        encodings = []
        for x_i in x:
            encodings.append(self.encoder.encode(self.tensor2img(x_i)))
        return torch.tensor(np.array(encodings))
    def get_neighbors(self, x):
        x_clip = self.get_encodings(x)
        b, _, block_size, _ = x.size()
        neighbors, _ = self.searcher.search_batched(x_clip, leaves_to_search=150, pre_reorder_num_neighbors=250)
        mat = []
        for neighbor in neighbors:
            mat.append(self.dset[np.int64(neighbor)])
        return torch.stack(mat).view(x.size(0), -1, block_size, block_size)
    def select_random_patches(self, image):
        _, _, height, width = image.shape


        patches = []

        for _ in range(self.patches):
            top_left_y = torch.randint(0, height - self.patch_size + 1, (1,))
            top_left_x = torch.randint(0, width - self.patch_size + 1, (1,))

            patch = image[:, :, top_left_y:(top_left_y + self.patch_size), top_left_x:(top_left_x + self.patch_size)]
            patches.append(torch.squeeze(patch, dim=0))

        return patches
    def dsetbuilder(self):
        """ Creates the D Set for this particular Dataset"""
        if os.path.exists(self.DSET_PATH):
            all_patches = torch.load(self.DSET_PATH)
        else:
            all_patches = []
            for x, _ in tqdm(self.full_dataloader, desc='Building DSET'):
                for patch in self.select_random_patches(x):
                    all_patches.append(self.encoder.encode(self.tensor2img(patch)))
            torch.save(torch.tensor(np.array(all_patches)), self.DSET_PATH)
        return all_patches

class CelebAHQ:
    def __init__(self, batch_size: int = 16, img_size = 256, device = None):
        """
        Wrapper to load, preprocess and deprocess CIFAR-10 dataset.
        Args:
            batch_size (int): Batch size, default: 16.
        """
        # self.classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
        # self.n_classes = 10

        self.batch_size = batch_size
        self.device = device
        self.mean = [0.5, 0.5, 0.5]
        self.std = [0.5, 0.5, 0.5]
        self.patch_size = img_size // 2
        self.patches = 5
        self.train_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(img_size),
            transforms.CenterCrop(img_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.Normalize(self.mean, self.std)
        ])
        self.encoder = SentenceTransformer('clip-ViT-B-32')

        self.val_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(256),
            transforms.CenterCrop(256),
            transforms.Normalize(self.mean, self.std)
        ])
        self.ROOT_PATH = '/hdd/avideep/blockLDM/data/celeba/'
        self.IMAGE_PATH = '/hdd/avideep/blockLDM/data/celeba/celeba_hq/images'
        self.DSET_PATH = '/hdd/avideep/blockLDM/data/celeba/celeba_hq/dset.pth'
        train_size = int(50000 * 0.9)
        # val size is 10000 in cifar10
        test_size = 50000 - train_size

        self.train_set_full = ImageFolder(self.IMAGE_PATH, self.train_transform)
        self.train_loader, self.val_loader = self.train_val_test_split(self.train_set_full, self.batch_size)
        self.full_dataloader = DataLoader(self.train_set_full, batch_size=1, num_workers=12, pin_memory=True)
        # invert normalization for tensor to image transform
        self.inv_normalize = transforms.Compose([
            transforms.Normalize(mean=0, std=[1./s for s in self.std]),
            transforms.Normalize(mean=[-m for m in self.mean], std=1.),
            lambda x: x*255
        ])
        self.dset = self.dsetbuilder()
        searcher_dir = '/hdd/avideep/blockLDM/data/celeba/celeba_hq/searcher/'
        if not os.path.exists(searcher_dir):
            self.searcher = scann.scann_ops_pybind.builder(self.dset / np.linalg.norm(self.dset, axis=1)[:, np.newaxis], 10, "dot_product").tree(num_leaves=2000, num_leaves_to_search=100, training_sample_size=250000).score_ah(2, anisotropic_quantization_threshold=0.2).reorder(100).build()
            print(f'Save trained searcher under "{searcher_dir}"')
            os.makedirs(searcher_dir, exist_ok=True)
            self.searcher.serialize(searcher_dir)
        else:
            print(f'Loading pre-trained searcher from {searcher_dir}')
            self.searcher = scann.scann_ops_pybind.load_searcher(searcher_dir).gpu()
            print('Finished loading searcher.')

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
    def get_encodings(self, x):
        encodings = []
        for x_i in x:
            encodings.append(self.encoder.encode(self.tensor2img(x_i)))
        return torch.tensor(np.array(encodings))
    def get_neighbors(self, x, block_size):
        x_clip = self.get_encodings(x)
        b = x.size(0)
        neighbors, _ = self.searcher.search_batched(x_clip.to(self.device), leaves_to_search=150, pre_reorder_num_neighbors=250)
        mat = []
        for neighbor in neighbors:
            mat.append(self.dset[np.int64(neighbor)])
        return torch.stack(mat).view(x.size(0), -1, block_size, block_size)
    def select_random_patches(self, image):
        _, _, height, width = image.shape


        patches = []

        for _ in range(self.patches):
            top_left_y = torch.randint(0, height - self.patch_size + 1, (1,))
            top_left_x = torch.randint(0, width - self.patch_size + 1, (1,))

            patch = image[:, :, top_left_y:(top_left_y + self.patch_size), top_left_x:(top_left_x + self.patch_size)]
            patches.append(torch.squeeze(patch, dim=0))

        return patches
    def dsetbuilder(self):
        """ Creates the D Set for this particular Dataset"""
        if os.path.exists(self.DSET_PATH):
            all_patches = torch.load(self.DSET_PATH)
        else:
            all_patches = []
            for x, _ in tqdm(self.full_dataloader, desc='Building DSET'):
                for patch in self.select_random_patches(x):
                    all_patches.append(self.encoder.encode(self.tensor2img(patch)))
            all_patches = torch.tensor(np.array(all_patches))
            torch.save(all_patches, self.DSET_PATH)
        return all_patches



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
