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
        # self.patch_size = self.data.img_size // block_factor
        self.DSET_PATH = self.data.ROOT_PATH + 'dset/{}/vqgan/whole_dset.pth'.format(data_name)
        self.k = k
        self.model = model
        self.device = device
        self.latent_dim, self.latent_size = self.get_latent_shape()[1], self.get_latent_shape()[2]
        self.latent_patch_size = self.latent_size // block_factor
        self.dset = self.dsetbuilder()
        searcher_dir = self.data.ROOT_PATH + 'dset/{}/vqgan/whole_searcher_k_{}/'.format(data_name, k)
        if not os.path.exists(searcher_dir):
            t_start = time.time()
            self.searcher = scann.scann_ops_pybind.builder(self.dset / np.linalg.norm(self.dset, axis=1)[:, np.newaxis].astype(np.float32), self.k, "dot_product").tree(num_leaves=2000, num_leaves_to_search=100, training_sample_size=250000).score_ah(2, anisotropic_quantization_threshold=0.2).reorder(100).build()
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
        return self.encode(torch.randn(1,3,self.data.img_size,self.data.img_size).to(self.device)).size()
    
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
    
    # @torch.no_grad()
    # def dsetbuilder(self):
    #     # x, _ = next(iter(self.data.full_dataloader))
    #     # print(self.model.encode(x.to(self.device)).shape)
    #     """ Creates the D Set for this particular Dataset"""
    #     if os.path.exists(self.DSET_PATH):
    #         all_patches = torch.load(self.DSET_PATH)
    #     else:
    #         print('DSET at {} does note exist. Building...'.format(self.DSET_PATH))
    #         all_patches = []
    #         for i in range(0, self.latent_size, self.latent_patch_size):
    #             for j in range (0, self.latent_size, self.latent_patch_size):
    #                 patches = []
    #                 for x, _ in tqdm(self.data.full_dataloader, desc='Building DSET'):
    #                     x = x.to(self.device)
    #                     z = self.encode(x)
    #                     patch = z[:, :, i:i+self.latent_patch_size, j:j+self.latent_patch_size]
    #                     patches.append(patch.cpu().detach())
    #                     del x, z, patch
    #                 all_patches.append(torch.cat(patches, dim=0).contiguous().view(len(self.data.full_dataloader.dataset), -1))
    #                 del patches
    #         all_patches = torch.stack(all_patches)
    #         torch.save(all_patches, self.DSET_PATH)
    #     print('DSET with shape: {} is ready!'.format(all_patches.shape))
    #     return all_patches
    @torch.no_grad()
    def dsetbuilder(self):
        """ Creates the D Set for this particular Dataset"""
        if os.path.exists(self.DSET_PATH):
            all_patches = torch.load(self.DSET_PATH)
        else:
            all_patches = []
            for x, _ in tqdm(self.data.full_dataloader, desc='Building DSET'):
                # batch_patches = self.get_random_patches(x)
                # clips = torch.stack([torch.tensor(self.encoder.encode(self.tensor2img(x_i))) for x_i in batch_patches])
                z = self.encode(x.to(device))
                all_patches.append(z.cpu().detach().contiguous().view(z.size(0), -1))
                # del batch_patches
                del z
            all_patches = torch.cat(all_patches, dim = 0)

            torch.save(all_patches.view(all_patches.size(0), -1), self.DSET_PATH)
        print('Dset of shape {} is ready!'.format(all_patches.shape))
        return all_patches
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DSET Building")
    parser.add_argument('--data', '-d', default='CelebA',
                        type=str, metavar='data', help='Dataset Name. Please enter CelebA, CelebAHQ, ImageNet100 or CIFAR10. Default: CelebA')
    parser.add_argument('--vqgan-path', default='checkpoints/vqgan/24-03-29_153956/best_model.pt',
                        metavar='PATH', help='Path to encoder/decoder model checkpoint (default: empty)')
    parser.add_argument('--vqgan-config', default='configs/vqgan_rgb.yaml',
                        metavar='PATH', help='Path to model config file (default: configs/vqgan_cifar10.yaml)')
    parser.add_argument('--gpus', default=0, type=int,
                    nargs='+', metavar='GPUS', help='If GPU(s) available, which GPU(s) to use for training.')
    parser.add_argument('--batch-size', default=16, metavar='N',
                    type=int, help='Mini-batch size (default: 16)')
    parser.add_argument('--k', default=10, metavar='N',
                    type=int, help='number of nearest neighbors.')
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
    dset = DSetBuilder(data, k=args.k, model=vqgan_model, device=device, block_factor=2)