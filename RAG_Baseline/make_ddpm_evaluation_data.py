import argparse
import os

import torch
import yaml

from tqdm import tqdm

from utils.visualization import tensor_to_image
from dataloader import PlantNet
from model import VQGANLight, VAE, IntroVAE
from model.ddpm.ddpm import DDPM
from model.unet.unet_light import UNetLight
import torchvision.transforms.functional as F
from utils.helpers import load_model_checkpoint
from dsetbuilder_clip import DSetBuilder
from dataloader import CelebA, CelebAHQ, CIFAR10, ImageNet100
# from: https://stackoverflow.com/questions/20554074/sklearn-omp-error-15-initializing-libiomp5md-dll-but-found-mk2iomp5md-dll-a
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

parser = argparse.ArgumentParser(description="PyTorch Second Stage Training")
parser.add_argument('--data', '-d', default='CelebA',
                        type=str, metavar='data', help='Dataset Name. Please enter CelebA, CelebAHQ, or CIFAR10. Default: CelebA')
parser.add_argument('--data-path', metavar='PATH', help='Path to root of the data')

parser.add_argument('--batch-size', default=16, metavar='N',
                    type=int, help='Mini-batch size (default: 64)')
parser.add_argument('--dset-batch-size', default=32, metavar='N',
                    type=int, help='Mini-batch size (default: 32)')
parser.add_argument('--image-size', default=224, metavar='N',
                    type=int, help='Size that images should be resized to before processing (default: 128)')
parser.add_argument('--block-size', default=14, metavar='N',
                    type=int, help='Size of the block that the image will be divided by.')
parser.add_argument('--k', default=20, metavar='N',
                    type=int, help='Number of nearest neighbors to search.')
parser.add_argument('--image-count', default=10,
                    type=int, help='number of images that should be generated for comparison')
parser.add_argument('--config', default='configs/ddpm_linear.yaml',
                    metavar='PATH', help='Path to model config file (default: configs/ddpm_linear.yaml)')
parser.add_argument('--unet-config', default='configs/unet.yaml',
                    metavar='PATH', help='Path to unet model config file (default: configs/unet.yaml)')
parser.add_argument('--load-ckpt-ddpm', default='checkpoints/second_stage/ddpm_linear/24-04-20_171311/ddpm/best_model.pt', metavar='PATH',
                    dest='load_checkpoint_ddpm', help='Load model checkpoint and continue training')
parser.add_argument('--load-ckpt-unet', default='checkpoints/second_stage/ddpm_linear/24-04-20_171311/unet/best_model.pt', metavar='PATH',
                    dest='load_checkpoint_unet', help='Load model checkpoint and continue training')
parser.add_argument('--vqgan-path', default='checkpoints/vqgan/24-03-18_151152/best_model.pt',
                    metavar='PATH', help='Path to encoder/decoder model checkpoint (default: empty)')
parser.add_argument('--vqgan-config', default='configs/vqgan_cifar10.yaml',
                    metavar='PATH', help='Path to model config file (default: configs/vqgan.yaml)')
parser.add_argument('--real-image-path', default='original',
                    metavar='PATH', help='Path to load samples from the source images into')
parser.add_argument('--gen-image-path', default='samples',
                    metavar='PATH', help='Path to generated images')
parser.add_argument('--use-prev-block', action='store_true',
                    help='Whether to condition the model with the previous block')
parser.add_argument('--gpus', default=0, type=int,
                    nargs='+', metavar='GPUS', help='If GPU(s) available, which GPU(s) to use for training.')
parser.add_argument('--sample_real', action='store_true',
                    help='If true, samples images from the ground truth imageset')
parser.add_argument('--sample_gen', action='store_true',
                    help='If true, samples images from the ddpm')
parser.add_argument('--data-config', default='configs/data_se.yaml',
                    metavar='PATH', help='Path to model config file (default: configs/data_se.yaml)')
# parser.add_argument('--vae-path', default='checkpoints/vae/introVAE/24-01-18_162139/best_model.pt',
#                     metavar='PATH', help='Path to encoder/decoder model checkpoint (default: empty)')
# parser.add_argument('--vae-config', default='configs/vae.yaml',
#                     metavar='PATH', help='Path to model config file (default: configs/vaeyaml)')



def main():
    args = parser.parse_args()
    for name, val in vars(args).items():
        print("{:<16}: {}".format(name, val))

    if args.sample_real:
        if args.real_image_path and not os.path.exists(args.real_image_path):
            os.makedirs(args.real_image_path)
        # GPU setup
        args.gpus = args.gpus if isinstance(args.gpus, list) else [args.gpus]
        if len(args.gpus) == 1:
            device = torch.device(f'cuda:{args.gpus[0]}' if torch.cuda.is_available() else 'cpu')
        else:
            raise ValueError('Currently multi-gpu training is not possible')
        print("{:<16}: {}".format('device', device))

        if args.data == 'CelebA':
            args.img_size = 64
            data = CelebA(args.batch_size)
        elif args.data == 'CIFAR10':
            data = CIFAR10(args.batch_size)
        elif args.data == 'ImageNet100':
            args.img_size = 224
            data = ImageNet100(batch_size = args.batch_size, dset_batch_size = args.dset_batch_size)
        else:
            data = CelebAHQ(args.batch_size, dset_batch_size= args.dset_batch_size, device=device)
        sample_images_real(data.val, args.image_count, args.real_image_path)

    if args.sample_gen:
        if args.sample_gen and args.gen_image_path and not os.path.exists(args.gen_image_path):
            os.makedirs(args.gen_image_path)

        # GPU setup
        args.gpus = args.gpus if isinstance(args.gpus, list) else [args.gpus]
        if len(args.gpus) == 1:
            device = torch.device(f'cuda:{args.gpus[0]}' if torch.cuda.is_available() else 'cpu')
        else:
            raise ValueError('Currently multi-gpu training is not possible')
        print("{:<16}: {}".format('device', device))
        if args.data == 'CelebA':
            args.img_size = 64
            data = CelebA(root= args.data_path, batch_size= args.batch_size)
        elif args.data == 'CIFAR10':
            data = CIFAR10(args.batch_size)
        elif args.data == 'ImageNet100':
            args.img_size = 224
            data = ImageNet100(root= args.data_path, batch_size = args.batch_size, dset_batch_size = args.dset_batch_size)
        else:
            data = CelebAHQ(args.batch_size, dset_batch_size= args.dset_batch_size, device=device)
        # read config file for model
        cfg = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)
        cfg_unet = yaml.load(open(args.unet_config, 'r'), Loader=yaml.Loader)
        cfg_vqgan = yaml.load(open(args.vqgan_config, 'r'), Loader=yaml.Loader)
        # cfg_vae = yaml.load(open(args.vae_config,'r'),Loader=yaml.Loader)

        vqgan_model = VQGANLight(**cfg_vqgan['model'])
        vqgan_model, _, _ = load_model_checkpoint(vqgan_model, args.vqgan_path, device)
        vqgan_model.to(device)
        global latent_dim
        latent_dim = cfg_vqgan['model']['latent_dim']
        if args.use_prev_block:
            cfg_unet['in_channels'] = (args.k + 2) * latent_dim # 2 because one if for the input latent representation of the current block and another is that for the previous block
        else:
            cfg_unet['in_channels'] = (args.k + 1) * latent_dim
        unet = UNetLight(**cfg_unet)
        unet, _, _ = load_model_checkpoint(unet, args.load_checkpoint_unet, device)
        unet.to(device)

        ddpm = DDPM(eps_model=unet, vae_model=vqgan_model, **cfg)
        ddpm, _, _ = load_model_checkpoint(ddpm, args.load_checkpoint_ddpm, device)
        ddpm.to(device)

        dset = DSetBuilder(data, args.k)

        # vae = IntroVAE(**cfg_vae['model'])
        # vae, _, _ = load_model_checkpoint(vae, args.vae_path, device)
        # vae.to(device)
        # global vae_latent_dim
        # vae_latent_dim = cfg_vae['model']['latent_dim']        

        sample_images_gen(ddpm, dset, args.image_count, args.gen_image_path, args, device)


def sample_images_real(data_loader, n_images, real_image_path):
    count = 0
    for x, _ in tqdm(data_loader, desc="sample_real_images"):
        for one_image in x:
            img = tensor_to_image(one_image)
            img.save(f"{real_image_path}/{count}.jpg")

            count += 1
            if count == n_images:
                return
# def sample_from_vae(n_images, model, device):
#     z = torch.randn(n_images, vae_latent_dim).to(device)
#     images = model.decode(z)
#     return images

@torch.no_grad()
def sample_images_gen(model, dset, n_images, image_path, args, device):
    model.eval()

    # we only want to sample x0 images
    sample_step = 0

    max_sample_size = 128
    step_count = 0

    while n_images > 0:
        if n_images >= max_sample_size:
            sample_size = max_sample_size
        else:
            sample_size = n_images
        x_query = dset.get_rand_queries(n_images)
        neighbor_ids = dset.get_neighbor_ids(x_query)
        shape = model.encode(torch.rand(n_images, args.image_channels, args.img_size, args.img_size).to(device)).size()
        neighbors = dset.get_neighbors(neighbor_ids, shape).to(device)
        images = model.sample(shape[2], neighbors, batch_size=n_images, channels=latent_dim)
        images = [img for img in images[0]]
        images = torch.stack(images)
        images = model.decode(images)

        for n, img in enumerate(images):
            img = tensor_to_image(img)
            img.save(f"{image_path}/{step_count}_{n}.jpg")

        n_images -= sample_size
        step_count += 1

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        raise e
