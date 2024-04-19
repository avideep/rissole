import os
import torch
import argparse
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.data import SubsetRandomSampler
import numpy as np
from PIL import Image
import torch.nn.functional as F
import os
import math
from argparse import ArgumentParser
import random
from glob import glob
from multiprocessing import cpu_count
import numpy as np
import scann
import gc
import torch
from einops import rearrange
from omegaconf import OmegaConf
from pytorch_lightning import seed_everything
from torch.utils.data import DataLoader, Sampler
from torch.utils.data.dataloader import default_collate
from tqdm import tqdm
import yaml
from model import VQGANLight
from utils.helpers import load_model_checkpoint, timer
import time
from dataloader import CelebA, CelebAHQ, CIFAR10, ImageNet100

class DSetBuilder:
    def __init__(self, data, k, model, device, block_factor = 2):
        data_name = data.__class__.__name__
        if data_name not in ['CelebA', 'CelebAHQ', 'CIFAR10', 'ImageNet100']:
            raise ValueError("Invalid input. Please enter CelebA, CelebAHQ, ImageNet100 or CIFAR10.")
        self.data = data
        self.mean = [0.5, 0.5, 0.5]
        self.std = [0.5, 0.5, 0.5]
        self.patch_size = self.data.img_size // block_factor
        self.DSET_PATH = '/hdd/avideep/blockLDM/data/dset/{}/vqgan/dset_f_{}.pth'.format(data_name, block_factor)
        self.k = k
        self.model = model
        self.device = device
        self.latent_dim, self.latent_size = self.get_latent_shape()[1], self.get_latent_shape()[2]
        self.latent_patch_size = self.latent_size // block_factor
        self.dset = self.dsetbuilder()
        searcher_dir = '/hdd/avideep/blockLDM/data/dset/{}/vqgan/searcher_k_{}_f_{}/'.format(data_name, k, block_factor)
        if not os.path.exists(searcher_dir):
            t_start = time.time()
            self.searcher = scann.scann_ops_pybind.builder(self.dset[0] / np.linalg.norm(self.dset[0], axis=1)[:, np.newaxis].astype(np.float32), self.k, "dot_product").tree(num_leaves=2000, num_leaves_to_search=100, training_sample_size=250000).score_ah(2, anisotropic_quantization_threshold=0.2).reorder(100).build()
            os.makedirs(searcher_dir, exist_ok=True)
            self.searcher.serialize(searcher_dir)
            elapsed_time = timer(t_start, time.time())
            print(f"Took {elapsed_time} to save trained searcher in {searcher_dir}")
        else:
            print(f'Loading pre-trained searcher from {searcher_dir}')
            self.searcher = scann.scann_ops_pybind.load_searcher(searcher_dir)
            print('Finished loading searcher.')
        
        
    @torch.no_grad()
    def encode(self, x: torch.Tensor):
        x = self.model.encode(x)
        x = self.model.quantize(x)
        return x
    
    def get_latent_shape(self):
        x = torch.randn(1,3,self.data.img_size,self.data.img_size).to(self.device)
        x = vqgan_model.encode(x)
        x = vqgan_model.quantize(x)
        return x.size()
    
    def get_neighbor_ids(self, x):
        # b = x.size(0)
        # x_vqgan = self.model.encode(x).view(b, -1)
        neighbors, _ = self.searcher.search_batched(x.cpu().numpy().astype(np.float32))
        return neighbors
    
    def get_rand_queries(self, n):
        indices = torch.randperm(self.dset.size(1))[:n]
        return self.dset[0, indices, :]
    
    def get_neighbors(self, neighbor_ids, position, block_size, b, latent_dim):
        mat = []
        for neighbor in neighbor_ids:
            mat.append(self.dset[position][np.int64(neighbor)])
        output = torch.stack(mat).view(b, self.k*latent_dim, block_size, block_size)
        # pad = (block_size - output.shape[-1])//2
        # padding = (pad, pad)
        # output = F.pad(output, padding, "constant", 0)
        return output
    
    @torch.no_grad()
    def dsetbuilder(self):
        # x, _ = next(iter(self.data.full_dataloader))
        # print(self.model.encode(x.to(self.device)).shape)
        """ Creates the D Set for this particular Dataset"""
        if os.path.exists(self.DSET_PATH):
            all_patches = torch.load(self.DSET_PATH)
        else:
            all_patches = []
            for i in range(0, self.latent_size, self.latent_patch_size):
                for j in range (0, self.latent_size, self.latent_patch_size):
                    patches = []
                    for x, _ in tqdm(self.data.full_dataloader, desc='Building DSET'):
                        x = x.to(self.device)
                        z = self.encode(x)
                        patch = z[:, :, i:i+self.latent_patch_size, j:j+self.latent_patch_size]
                        patches.append(patch.cpu().detach())
                        del x, z, patch
                    all_patches.append(torch.cat(patches, dim=0).view(len(self.data.full_dataloader.dataset), -1))
            all_patches = torch.stack(all_patches)
            torch.save(all_patches, self.DSET_PATH)
        print('DSET with shape: {} is ready!'.format(all_patches.shape))
        return all_patches
    
class ClassDSetBuilder:
    def __init__(self, data, k, model, device):
        data_name = data.__class__.__name__
        if data_name not in ['CelebA', 'CelebAHQ', 'CIFAR10', 'ImageNet100']:
            raise ValueError("Invalid input. Please enter CelebA, CelebAHQ, ImageNet100 or CIFAR10.")
        self.data = data
        self.mean = [0.5, 0.5, 0.5]
        self.std = [0.5, 0.5, 0.5]
        self.patch_size = self.data.img_size // 2
        self.DSET_PATH = '/hdd/avideep/blockLDM/data/dset/' + data_name + '/vqgan/classy/dset.pth'
        self.k = k
        self.model = model
        self.device = device
        self.classes = self.data.train_set_full.classes
        self.dset = self.dsetbuilder()
        self.searchers = []
        for class_idx in range(len(self.classes)):
            searcher_dir = '/hdd/avideep/blockLDM/data/dset/' + data_name + '/vqgan/classy/searcher_' + str(k) + '_' + str(class_idx) + '/'
            if not os.path.exists(searcher_dir):
                t_start = time.time()
                searcher = scann.scann_ops_pybind.builder(self.dset[class_idx][0] / np.linalg.norm(self.dset[class_idx][0], axis=1)[:, np.newaxis].astype(np.float32), self.k, "dot_product").tree(num_leaves= 200, num_leaves_to_search=100, training_sample_size=25000).score_ah(2, anisotropic_quantization_threshold=0.2).reorder(100).build()
                os.makedirs(searcher_dir, exist_ok=True)
                searcher.serialize(searcher_dir)
                elapsed_time = timer(t_start, time.time())
                print(f"Took {elapsed_time} to save trained searcher in {searcher_dir}")
                self.searchers.append(searcher)
            else:
                print(f'Loading pre-trained searcher from {searcher_dir}')
                searcher = scann.scann_ops_pybind.load_searcher(searcher_dir)
                self.searchers.append(searcher)
        print('Finished loading searchers.')
    @torch.no_grad()
    def encode(self, x: torch.Tensor):
        x = self.model.encode(x)
        x = self.model.quantize(x)
        return x

    def get_neighbor_ids(self, x, c):
        neighbors = [] 
        for data, class_idx in zip(x,c):
            neighbor, _ = self.searchers[class_idx].search(data.cpu().numpy().astype(np.float32))
            neighbors.append([neighbor, class_idx])
        return neighbors
    
    def get_rand_queries(self, n):
        classes = np.array(random.choices(list(range(len(self.classes))), k = n))
        queries = torch.stack([self.dset[cl][0, random.randint(0, len(self.dset[cl][0])-1), :] for cl in classes])
        return queries, torch.tensor(classes)
    
    def get_neighbors(self, neighbor_ids, position, block_size, b, latent_dim):
        mat = []
        for neighbor, class_idx in neighbor_ids:
            mat.append(self.dset[class_idx.item()][position][np.int64(neighbor)])
        output = torch.stack(mat).view(b, self.k*latent_dim, block_size, block_size)
        # pad = (block_size - output.shape[-1])//2
        # padding = (pad, pad)
        # output = F.pad(output, padding, "constant", 0)
        return output
    
    def get_num_blocks(self):
        return (self.data.img_size//self.patch_size)**2
    
    @torch.no_grad()
    def dsetbuilder(self):
        # x, _ = next(iter(self.data.full_dataloader))
        # print(self.model.encode(x.to(self.device)).shape)
        """ Creates the D Set for this particular Dataset"""
        if os.path.exists(self.DSET_PATH):
            all_patches = torch.load(self.DSET_PATH)
        else:
            position = 0
            all_patches = {i: [[] for i in range(self.get_num_blocks())] for i in range(len(self.classes))}
            for i in range(0, self.data.img_size, self.patch_size):
                for j in range (0, self.data.img_size, self.patch_size):
                    for x, c in tqdm(self.data.full_dataloader, desc='Building DSET'):
                        # print(c)
                        # print(self.classes)
                        x = x.to(self.device)
                        patch = x[:, :, i:i+self.patch_size, j:j+self.patch_size]
                        patch_encoded = self.model.encode(patch).cpu().detach().view(x.size(0),-1)
                        for i in range(x.size(0)):
                            all_patches[c[i].item()][position].append(patch_encoded[i])
                    position += 1
            for cl_name, data in all_patches.items():
                for i in range(self.get_num_blocks()):
                    data[i] = torch.stack(data[i])
                all_patches[cl_name] = torch.stack(data)
            print(all_patches.keys())
            print(all_patches[0].shape)
            torch.save(all_patches, self.DSET_PATH)
        # print('DSET with shape: {} is ready!'.format(all_patches.shape))
        return all_patches
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DSET Building")
    parser.add_argument('--data', '-d', default='CelebA',
                        type=str, metavar='data', help='Dataset Name. Please enter CelebA, CelebAHQ, ImageNet100 or CIFAR10. Default: CelebA')
    parser.add_argument('--vqgan-path', default='checkpoints/vqgan/24-03-18_151152/best_model.pt',
                        metavar='PATH', help='Path to encoder/decoder model checkpoint (default: empty)')
    parser.add_argument('--vqgan-config', default='configs/vqgan_cifar10.yaml',
                        metavar='PATH', help='Path to model config file (default: configs/vqgan_cifar10.yaml)')
    parser.add_argument('--gpus', default=0, type=int,
                    nargs='+', metavar='GPUS', help='If GPU(s) available, which GPU(s) to use for training.')
    parser.add_argument('--batch-size', default=16, metavar='N',
                    type=int, help='Mini-batch size (default: 16)')
    parser.add_argument('--dset-batch-size', default=64, metavar='N',
                    type=int, help='Mini-batch size (default: 32)')
    args = parser.parse_args()
    cfg_vqgan = yaml.load(open(args.vqgan_config, 'r'), Loader=yaml.Loader)
    # setup GPU
    args.gpus = args.gpus if isinstance(args.gpus, list) else [args.gpus]
    if len(args.gpus) == 1:
        device = torch.device(f'cuda:{args.gpus[0]}' if torch.cuda.is_available() else 'cpu')
    else:
        raise ValueError('Currently multi-gpu training is not possible')
    vqgan_model = VQGANLight(**cfg_vqgan['model'])
    vqgan_model, _, _ = load_model_checkpoint(vqgan_model, args.vqgan_path, device)
    vqgan_model.to(device)

    if args.data == 'CelebA':
        data = CelebA(batch_size = args.batch_size, dset_batch_size = args.dset_batch_size)
    elif args.data == 'CelebAHQ':
        data = CelebAHQ(batch_size = args.batch_size, dset_batch_size = args.dset_batch_size)
    elif args.data == 'ImageNet100':
        data = ImageNet100(batch_size = args.batch_size, dset_batch_size = args.dset_batch_size)
    else:
        data = CIFAR10(batch_size = args.batch_size, dset_batch_size = args.dset_batch_size)
    dset = ClassDSetBuilder(data, k=20, model=vqgan_model, device=device)