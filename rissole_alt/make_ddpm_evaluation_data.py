import argparse
import os

import torch
import yaml

from tqdm import tqdm
import random
import string

from utils.visualization import tensor_to_image
from dataloader import PlantNet
from model import VQGANLight, VAE, IntroVAE
from model.ddpm.ddpm import DDPM
from model.unet.unet_light import UNetLight
import torchvision.transforms.functional as F
from utils.helpers import load_model_checkpoint
from dsetbuilder_vqgan import DSetBuilder
from dataloader import CelebA, CelebAHQ, CIFAR10, ImageNet100
# from: https://stackoverflow.com/questions/20554074/sklearn-omp-error-15-initializing-libiomp5md-dll-but-found-mk2iomp5md-dll-a
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

parser = argparse.ArgumentParser(description="PyTorch Second Stage Training")
parser.add_argument('--data', '-d', default='CelebA',
                        type=str, metavar='data', help='Dataset Name. Please enter CelebA, CelebAHQ, or CIFAR10. Default: CelebA')
parser.add_argument('--batch-size', default=16, metavar='N',
                    type=int, help='Mini-batch size (default: 64)')
parser.add_argument('--dset-batch-size', default=32, metavar='N',
                    type=int, help='Mini-batch size (default: 32)')
parser.add_argument('--image-size', default=224, metavar='N',
                    type=int, help='Size that images should be resized to before processing (default: 128)')
parser.add_argument('--image-channels', default=3, metavar='N',
                    type=int, help='Number of image channels (default: 3)')
parser.add_argument('--data-path', default= '/hdd/avideep/blockLDM/data/', metavar='PATH', help='Path to root of the data')
parser.add_argument('--block-factor', default=4, metavar='N',
                    type=int, help='Size of the block that the image will be divided by.')
parser.add_argument('--k', default=20, metavar='N',
                    type=int, help='Number of nearest neighbors to search.')
parser.add_argument('--image-count', default=100,
                    type=int, help='number of images that should be generated for comparison')
parser.add_argument('--config', default='configs/ddpm_linear.yaml',
                    metavar='PATH', help='Path to model config file (default: configs/ddpm_linear.yaml)')
parser.add_argument('--unet-config', default='configs/unet.yaml',
                    metavar='PATH', help='Path to unet model config file (default: configs/unet.yaml)')
parser.add_argument('--load-ckpt-ddpm', default='checkpoints/second_stage/ddpm_linear/24-05-12_134544/ddpm/best_model.pt', metavar='PATH',
                    dest='load_checkpoint_ddpm', help='Load model checkpoint and continue training')
parser.add_argument('--load-ckpt-unet', default='checkpoints/second_stage/ddpm_linear/24-05-12_134544/unet/best_model.pt', metavar='PATH',
                    dest='load_checkpoint_unet', help='Load model checkpoint and continue training')
parser.add_argument('--vqgan-path', default='checkpoints/vqgan/24-03-18_151152/best_model.pt',
                    metavar='PATH', help='Path to encoder/decoder model checkpoint (default: empty)')
parser.add_argument('--vqgan-config', default='configs/vqgan_cifar10.yaml',
                    metavar='PATH', help='Path to model config file (default: configs/vqgan.yaml)')
parser.add_argument('--real-image-path', default='original',
                    metavar='PATH', help='Path to load samples from the source images into')
parser.add_argument('--gen-image-path', default='samples/new_method_blocks_4/',
                    metavar='PATH', help='Path to generated images')
parser.add_argument('--use-prev-block', action='store_true',
                    help='Whether to condition the model with the previous block')
parser.add_argument('--use-rag', action='store_true',
                    help='Whether to condition on retrieved neighbors from an external memory')
parser.add_argument('--gpus', default=0, type=int,
                    nargs='+', metavar='GPUS', help='If GPU(s) available, which GPU(s) to use for training.')
parser.add_argument('--sample_real', action='store_true',
                    help='If true, samples images from the ground truth imageset')
parser.add_argument('--sample_gen', action='store_true',
                    help='If true, samples images from the ddpm')
parser.add_argument('--use-pos', action='store_true',
                    help='If true, condition the images with positional information')
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
        # if args.real_image_path and not os.path.exists(args.real_image_path):
        #     os.makedirs(args.real_image_path)
        # GPU setup
        args.gpus = args.gpus if isinstance(args.gpus, list) else [args.gpus]
        if len(args.gpus) == 1:
            device = torch.device(f'cuda:{args.gpus[0]}' if torch.cuda.is_available() else 'cpu')
        else:
            raise ValueError('Currently multi-gpu training is not possible')
        print("{:<16}: {}".format('device', device))

        if args.data == 'CelebA':
            if args.block_factor == 3:
                args.unet_config = 'configs/unet_f_3.yaml'
                args.img_size = 60
            else:
                args.img_size = 64
            data = CelebA(root= args.data_path, batch_size= args.batch_size, img_size=args.img_size)
            args.real_image_path += '/celeba/'
        elif args.data == 'CIFAR10':
            data = CIFAR10(args.batch_size)
        elif args.data == 'ImageNet100':
            args.vqgan_config = 'configs/vqgan_rgb.yaml'
            args.vqgan_path = 'checkpoints/vqgan/24-03-29_153956/best_model.pt'
            args.unet_config = 'configs/unet_imagenet100.yaml'
            if args.block_factor == 3:
                args.img_size = 216
            else:
                args.img_size = 224
            data = ImageNet100(root= args.data_path, batch_size = args.batch_size, dset_batch_size = args.dset_batch_size, img_size= args.img_size)
            args.real_image_path += '/imagenet100/'
        else:
            data = CelebAHQ(args.batch_size, dset_batch_size= args.dset_batch_size, device=device)
    # read config file for model
        if args.real_image_path and not os.path.exists(args.real_image_path):
            os.makedirs(args.real_image_path)
        cfg_vqgan = yaml.load(open(args.vqgan_config, 'r'), Loader=yaml.Loader)
        vqgan_model = VQGANLight(**cfg_vqgan['model'])
        vqgan_model, _, _ = load_model_checkpoint(vqgan_model, args.vqgan_path, device)
        vqgan_model.to(device)
        sample_images_real(data.val, args.image_count, vqgan_model, device, args.real_image_path)

    if args.sample_gen:
        if args.gen_image_path and not os.path.exists(args.gen_image_path):
            os.makedirs(args.gen_image_path)

        # GPU setup
        args.gpus = args.gpus if isinstance(args.gpus, list) else [args.gpus]
        if len(args.gpus) == 1:
            device = torch.device(f'cuda:{args.gpus[0]}' if torch.cuda.is_available() else 'cpu')
        else:
            raise ValueError('Currently multi-gpu training is not possible')
        print("{:<16}: {}".format('device', device))
        if args.data == 'CelebA':
            if args.block_factor == 3:
                args.unet_config = 'configs/unet_f_3.yaml'
                args.img_size = 60
            else:
                args.img_size = 64
            data = CelebA(root= args.data_path, batch_size= args.batch_size, img_size=args.img_size)
        elif args.data == 'CIFAR10':
            data = CIFAR10(args.batch_size)
        elif args.data == 'ImageNet100':
            args.vqgan_config = 'configs/vqgan_rgb.yaml'
            args.vqgan_path = 'checkpoints/vqgan/24-03-29_153956/best_model.pt'
            args.unet_config = 'configs/unet_imagenet100.yaml'
            if args.block_factor == 3:
                args.img_size = 216
            else:
                args.img_size = 224
            data = ImageNet100(root= args.data_path, batch_size = args.batch_size, dset_batch_size = args.dset_batch_size, img_size= args.img_size)
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
        # if args.use_prev_block:
        #     cfg_unet['in_channels'] = (args.k + 2) * latent_dim # 2 because one if for the input latent representation of the current block and another is that for the previous block
        # else:
        cfg_unet['cond_emb_dim'] = args.k * latent_dim

        unet = UNetLight(**cfg_unet)
        unet, _, _ = load_model_checkpoint(unet, args.load_checkpoint_unet, device)
        unet.to(device)

        ddpm = DDPM(eps_model=unet, vae_model=vqgan_model, **cfg)
        ddpm, _, _ = load_model_checkpoint(ddpm, args.load_checkpoint_ddpm, device)
        ddpm.to(device)

        if args.use_rag:
            print('Using RAG...')
            dset = DSetBuilder(data, args.k, vqgan_model, device, block_factor=args.block_factor)
        else:
            print('Not Using RAG...')
            dset = None
        # dset = torch.zeros(args.batch_size,  args.k * latent_dim, block_size, block_size).to(device)

        # dset = DSetBuilder(data, args.k, vqgan_model, device, block_factor=args.block_factor)

        # vae = IntroVAE(**cfg_vae['model'])
        # vae, _, _ = load_model_checkpoint(vae, args.vae_path, device)
        # vae.to(device)
        # global vae_latent_dim
        # vae_latent_dim = cfg_vae['model']['latent_dim']        
        block_size = get_block_size(args, vqgan_model, device)
        sample_images_gen(ddpm, dset, block_size, args.image_count, args.gen_image_path, args.img_size, args.use_rag, args.use_pos, args.k, device)

def get_block_size(args, vqgan_model, device):
    x = torch.rand(1, args.image_channels, args.img_size, args.img_size).to(device)
    x = vqgan_model.encode(x)
    x = vqgan_model.quantize(x)
    return x.size(2) // args.block_factor

def sample_images_real(data_loader, n_images, vqgan_model, device, real_image_path):
    count = 0
    real_image_path += 'vqgan'
    for x, _ in tqdm(data_loader, desc="sample_real_images"):
        x = vqgan_model.encode(x.to(device))
        x = vqgan_model.quantize(x)
        x = vqgan_model.decode(x)
        
        for one_image in x.cpu():
            img = tensor_to_image(one_image)
            img.save(f"{real_image_path}/{get_random_filename()}.jpg")

            count += 1
            if count == n_images:
                return
# def sample_from_vae(n_images, model, device):
#     z = torch.randn(n_images, vae_latent_dim).to(device)
#     images = model.decode(z)
#     return images

def get_random_filename():
    # Generate a random string of 10 characters
    return ''.join(random.choice(string.ascii_letters + string.digits) for _ in range(10))
@torch.no_grad()
def sample_images_gen(model, dset, block_size, n_images, image_path, image_size, use_rag, use_pos, nn, device):
    model.eval()

    # we only want to sample x0 images
    sample_step = 0

    max_sample_size = 50
    step_count = 0

    while n_images > 0:
        if n_images >= max_sample_size:
            sample_size = max_sample_size
        else:
            sample_size = n_images
        images_decoded = images = [0]*model.n_steps
        channels = 3
        img = torch.randn((sample_size, channels, image_size, image_size), device=device)
        img = model.encode(img)
        for i in range(len(images)):
            images[i] = img
        if use_rag:
            x_query = dset.get_rand_queries(sample_size)
            neighbor_ids = dset.get_neighbor_ids(x_query)
        low_res_cond = None
        position = 0
        for i in range(0, img.shape[-1], block_size):
            for j in range(0, img.shape[-1], block_size):

                block_pos = torch.full((sample_size,),position, dtype=torch.int64).to(device) if use_pos else None
                neighbors = dset.get_neighbors(neighbor_ids, position).to(device) if use_rag else torch.rand(sample_size,  nn * latent_dim, block_size, block_size).to(device)
                curr_block = model.sample(block_size, neighbors, block_pos, low_res_cond, batch_size=sample_size, channels=latent_dim)
                position += 1
                for k in range(len(curr_block)):
                    images[k][:, :, i:i+block_size, j:j+block_size] = curr_block[k]
        for k in range(len(images)):
            images_decoded[k] = model.decode(images[k])
        images = [img for img in images_decoded[0]]
        images = torch.stack(images)
        os.makedirs(image_path, exist_ok=True)
        for n, img in enumerate(images):
            img = tensor_to_image(img)
            img.save(f"{image_path}/{get_random_filename()}_{n}.jpg")

        n_images -= sample_size
        step_count += 1
if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        raise e
