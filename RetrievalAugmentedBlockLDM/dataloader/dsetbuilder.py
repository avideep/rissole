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
from argparse import ArgumentParser
from glob import glob
from multiprocessing import cpu_count
import numpy as np
import scann
import torch
from einops import rearrange
from omegaconf import OmegaConf
from pytorch_lightning import seed_everything
from torch.utils.data import DataLoader, Sampler
from torch.utils.data.dataloader import default_collate
from tqdm import tqdm
from sentence_transformers import SentenceTransformer, util

class DSetBuilder:
    def __init__(self, data, k):
        data_name = data.__class__.__name__
        if data_name not in ['CelebA', 'CelebAHQ', 'CIFAR10']:
            raise ValueError("Invalid input. Please enter CelebA, CelebAHQ, or CIFAR10.")
        self.data = data
        self.mean = [0.5, 0.5, 0.5]
        self.std = [0.5, 0.5, 0.5]
        self.patch_size = self.data.img_size // 2
        self.DSET_PATH = '/hdd/avideep/blockLDM/data/dset/' + data_name + '/dset.pth'
        self.encoder = SentenceTransformer('clip-ViT-B-32')
        self.inv_normalize = transforms.Compose([
                                transforms.Normalize(mean=0, std=[1./s for s in self.std]),
                                transforms.Normalize(mean=[-m for m in self.mean], std=1.),
                                lambda x: x*255])
        self.dset = self.dsetbuilder()
        self.k = k
        searcher_dir = '/hdd/avideep/blockLDM/data/' + data_name + '/searcher/'
        if not os.path.exists(searcher_dir):
            self.searcher = scann.scann_ops_pybind.builder(self.dset[0] / np.linalg.norm(self.dset[0], axis=1)[:, np.newaxis], self.k, "dot_product").tree(num_leaves=2000, num_leaves_to_search=100, training_sample_size=250000).score_ah(2, anisotropic_quantization_threshold=0.2).reorder(100).build()
            print(f'Save trained searcher under "{searcher_dir}"')
            os.makedirs(searcher_dir, exist_ok=True)
            self.searcher.serialize(searcher_dir)
        else:
            print(f'Loading pre-trained searcher from {searcher_dir}')
            self.searcher = scann.scann_ops_pybind.load_searcher(searcher_dir).gpu()
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

    def get_neighbors(self, x, position, block_size):
        x_clip = torch.tensor(np.array([self.encoder.encode(self.tensor2img(x_i)) for x_i in x]))
        b, _, _, _ = x.size()
        neighbors, _ = self.searcher.search_batched(x_clip)
        mat = []
        for neighbor in neighbors:
            mat.append(self.dset[position][np.int64(neighbor)])
        output = torch.stack(mat).view(b, self.k, block_size, -1)
        pad = (block_size - output.shape[-1])//2
        padding = (pad, pad)
        output = F.pad(output, padding, "constant", 0)

    def dsetbuilder(self):
        """ Creates the D Set for this particular Dataset"""
        if os.path.exists(self.DSET_PATH):
            all_patches = torch.load(self.DSET_PATH)
        else:
            all_patches = []
            for i in range(0, self.data.img_size, self.patch_size):
                for j in range (0, self.data.img_size, self.patch_size):
                    patches = []
                    for x, _ in tqdm(self.data.full_dataloader, desc='Building DSET'):
                        patch = x[:, :, i:i+self.patch_size, j:j+self.patch_size]
                        patches.append(self.encoder.encode(self.tensor2img(torch.squeeze(patch, dim = 0))))
                    all_patches.append(torch.tensor(np.array(patches)))
            
            torch.save(torch.stack(all_patches), self.DSET_PATH)
        return all_patches
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DSET Building")
    parser.add_argument('--data', '-d', default='CelebA',
                        type=str, metavar='data', help='Dataset Name. Please enter CelebA, CelebAHQ, or CIFAR10. Default: CelebA')
    args = parser.parse_args()
    dset = DSetBuilder(data=args.data)