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

from sentence_transformers import SentenceTransformer
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
    def __init__(self, data, k):
        data_name = data.__class__.__name__
        if data_name not in ['CelebA', 'CelebAHQ', 'CIFAR10', 'ImageNet100']:
            raise ValueError("Invalid input. Please enter CelebA, CelebAHQ, or CIFAR10 or ImageNet100.")
        self.data = data
        self.mean = [0.5, 0.5, 0.5]
        self.std = [0.5, 0.5, 0.5]
        self.patch_size = self.data.img_size // 2
        self.num_patches = 4
        self.num_channels = 3
        self.DSET_PATH = '/hdd/avideep/blockLDM/data/baseline_dset/' + data_name + '/dset_clip.pth'
        self.encoder = SentenceTransformer('clip-ViT-B-32')
        self.inv_normalize = transforms.Compose([
                                transforms.Normalize(mean=0, std=[1./s for s in self.std]),
                                transforms.Normalize(mean=[-m for m in self.mean], std=1.),
                                lambda x: x*255])
        self.dset = self.dsetbuilder()
        self.k = k
        searcher_dir = '/hdd/avideep/blockLDM/data/baseline_dset/' + data_name + '/searcher_clip/'
        if not os.path.exists(searcher_dir):
            self.searcher = scann.scann_ops_pybind.builder(self.dset / np.linalg.norm(self.dset, axis=1)[:, np.newaxis], self.k, "dot_product").tree(num_leaves=2000, num_leaves_to_search=100, training_sample_size=250000).score_ah(2, anisotropic_quantization_threshold=0.2).reorder(100).build()
            print(f'Save trained searcher under "{searcher_dir}"')
            os.makedirs(searcher_dir, exist_ok=True)
            self.searcher.serialize(searcher_dir)
        else:
            print(f'Loading pre-trained searcher from {searcher_dir}')
            self.searcher = scann.scann_ops_pybind.load_searcher(searcher_dir)
            print('Finished loading searcher.')

    def tensor2img(self, tensor):
        """ Convert torch.Tensor to PIL image. """
        n_channels = tensor.shape[0]

        img = tensor.detach().cpu()
        img = self.inv_normalize(img)

        if n_channels > 1:
            return Image.fromarray(img.permute(1, 2, 0).numpy().astype('uint8')).convert("RGB")
        else:
            return Image.fromarray(img[0].numpy()).convert("L")
    def get_rand_queries(self, n):
        return  self.dset[torch.randperm(self.dset.size(0))[:n]].view(n, 3, self.patch_size, self.patch_size)
    
    def get_neighbor_ids(self, x):
        # x_clip = torch.tensor(np.array([self.encoder.encode(self.tensor2img(x_i)) for x_i in x]))
        x = transforms.functional.resize(x, [self.patch_size], antialias = True)
        neighbors, _ = self.searcher.search_batched(x.contiguous().view(x.size(0), -1))
        return neighbors
    def get_neighbors(self, neighbor_ids, shape):
        # mat = [torch.stack([torch.tensor(self.encoder.encode(self.tensor2img(self.dset[np.int64(one_neighbor)].view(self.num_channels, self.patch_size, self.patch_size))) for one_neighbor in neighbor])) for neighbor in neighbor_ids]
        batch_size, num_channels, img_size, _ = shape
        mat = []
        for neighbor in neighbor_ids:
            neighbor_images = []
            for one_neighbor in neighbor:
                image = self.dset[np.int64(one_neighbor)].view(self.num_channels, self.patch_size, self.patch_size)
                encoded_image = self.encoder.encode(self.tensor2img(image))
                neighbor_images.append(torch.tensor(encoded_image))
            mat.append(torch.stack(neighbor_images))
        
        output = torch.stack(mat)
        output = output.view(batch_size, -1, img_size, img_size)
        # pad = (img_size - output.shape[-1])//2
        # padding = (pad, pad)
        # output = F.pad(output, padding, "constant", 0)
        return output
    def get_random_patches(self, images):
        batch_patches = []
        for image in images:
            for _ in range(self.num_patches):
                # Randomly select top-left corner coordinates for the patch
                top = random.randint(0, image.shape[1] - self.patch_size)
                left = random.randint(0, image.shape[2] - self.patch_size)

                # Extract the patch
                patch = image[:, top:top+self.patch_size, left:left+self.patch_size]
                batch_patches.append(patch)
                del patch
        return torch.stack(batch_patches)
    def get_clip_embeddings(self, x):
        return 
    def dsetbuilder(self):
        """ Creates the D Set for this particular Dataset"""
        if os.path.exists(self.DSET_PATH):
            all_patches = torch.load(self.DSET_PATH)
        else:
            all_patches = []
            for x, _ in tqdm(self.data.full_dataloader, desc='Building DSET'):
                batch_patches = self.get_random_patches(x)
                print(batch_patches.shape)
                clips = torch.stack([torch.tensor(self.encoder.encode(self.tensor2img(x_i))) for x_i in batch_patches])
                all_patches.append(clips)
                del batch_patches
                del clips
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
    parser.add_argument('--dset-batch-size', default=32, metavar='N',
                    type=int, help='Mini-batch size (default: 32)')
    args = parser.parse_args()
    # cfg_vqgan = yaml.load(open(args.vqgan_config, 'r'), Loader=yaml.Loader)
    # # setup GPU
    # args.gpus = args.gpus if isinstance(args.gpus, list) else [args.gpus]
    # if len(args.gpus) == 1:
    #     device = torch.device(f'cuda:{args.gpus[0]}' if torch.cuda.is_available() else 'cpu')
    # else:
    #     raise ValueError('Currently multi-gpu training is not possible')
    # vqgan_model = VQGANLight(**cfg_vqgan['model'])
    # vqgan_model, _, _ = load_model_checkpoint(vqgan_model, args.vqgan_path, device)
    # vqgan_model.to(device)
    # x = torch.randn(16,3,112,112).to(device)
    # x = vqgan_model.encode(x)
    # x = vqgan_model.quantize(x)

    if args.data == 'CelebA':
        data = CelebA(batch_size = args.batch_size, dset_batch_size = args.dset_batch_size)
    elif args.data == 'CelebAHQ':
        data = CelebAHQ(batch_size = args.batch_size, dset_batch_size = args.dset_batch_size)
    elif args.data == 'ImageNet100':
        data = ImageNet100(batch_size = args.batch_size, dset_batch_size = args.dset_batch_size)
    else:
        data = CIFAR10(batch_size = args.batch_size, dset_batch_size = args.dset_batch_size)
    dset = DSetBuilder(data, k=20)
    x = torch.rand(16, 3, 224, 224)
    n_ids = dset.get_neighbor_ids(x)
    ns = dset.get_neighbors(n_ids, x.size())
