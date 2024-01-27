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

# from: https://stackoverflow.com/questions/20554074/sklearn-omp-error-15-initializing-libiomp5md-dll-but-found-mk2iomp5md-dll-a
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

parser = argparse.ArgumentParser(description="PyTorch Second Stage Training")
parser.add_argument('--image-size', default=256, metavar='N',
                    type=int, help='Size that images should be resized to before processing (default: 128)')
parser.add_argument('--block-size', default=32, metavar='N',
                    type=int, help='Size of the block that the image will be divided by.')
parser.add_argument('--image-count', default=16,
                    type=int, help='number of images that should be generated for comparison')
parser.add_argument('--config', default='configs/ddpm_linear.yaml',
                    metavar='PATH', help='Path to model config file (default: configs/ddpm_linear.yaml)')
parser.add_argument('--unet-config', default='configs/unet.yaml',
                    metavar='PATH', help='Path to unet model config file (default: configs/unet.yaml)')
parser.add_argument('--load-ckpt-ddpm', default='checkpoints/second_stage/ddpm_linear/24-01-26_155614/ddpm/best_model.pt', metavar='PATH',
                    dest='load_checkpoint_ddpm', help='Load model checkpoint and continue training')
parser.add_argument('--load-ckpt-unet', default='checkpoints/second_stage/ddpm_linear/24-01-26_155614/unet/best_model.pt', metavar='PATH',
                    dest='load_checkpoint_unet', help='Load model checkpoint and continue training')
parser.add_argument('--vqgan-path', default='checkpoints/vqgan/24-01-17_130119/best_model.pt',
                    metavar='PATH', help='Path to encoder/decoder model checkpoint (default: empty)')
parser.add_argument('--vqgan-config', default='configs/vqgan_cifar10.yaml',
                    metavar='PATH', help='Path to model config file (default: configs/vqgan.yaml)')
parser.add_argument('--real-image-path', default='',
                    metavar='PATH', help='Path to load samples from the source images into')
parser.add_argument('--gen-image-path', default='samples',
                    metavar='PATH', help='Path to generated images')
parser.add_argument('--gpus', default=0, type=int,
                    nargs='+', metavar='GPUS', help='If GPU(s) available, which GPU(s) to use for training.')
parser.add_argument('--sample_real', action='store_true',
                    help='If true, samples images from the ground truth imageset')
parser.add_argument('--sample_gen', action='store_true',
                    help='If true, samples images from the ddpm')
parser.add_argument('--data-config', default='configs/data_se.yaml',
                    metavar='PATH', help='Path to model config file (default: configs/data_se.yaml)')
parser.add_argument('--vae-path', default='checkpoints/vae/introVAE/24-01-18_162139/best_model.pt',
                    metavar='PATH', help='Path to encoder/decoder model checkpoint (default: empty)')
parser.add_argument('--vae-config', default='configs/vae.yaml',
                    metavar='PATH', help='Path to model config file (default: configs/vaeyaml)')



def main():
    args = parser.parse_args()
    for name, val in vars(args).items():
        print("{:<16}: {}".format(name, val))

    if args.sample_real:
        if args.real_image_path and not os.path.exists(args.real_image_path):
            os.makedirs(args.real_image_path)

        data_cfg = yaml.load(open(args.data_config, 'r'), Loader=yaml.Loader)
        data = PlantNet(**data_cfg, batch_size=1, image_size=args.image_size,
                        num_workers=1)
        sample_images_real(data.test, args.image_count, args.real_image_path)

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

        # read config file for model
        cfg = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)
        cfg_unet = yaml.load(open(args.unet_config, 'r'), Loader=yaml.Loader)
        cfg_vqgan = yaml.load(open(args.vqgan_config, 'r'), Loader=yaml.Loader)
        cfg_vae = yaml.load(open(args.vae_config,'r'),Loader=yaml.Loader)

        vqgan_model = VQGANLight(**cfg_vqgan['model'])
        vqgan_model, _, _ = load_model_checkpoint(vqgan_model, args.vqgan_path, device)
        vqgan_model.to(device)

        unet = UNetLight(**cfg_unet)
        unet, _, _ = load_model_checkpoint(unet, args.load_checkpoint_unet, device)
        unet.to(device)

        ddpm = DDPM(eps_model=unet, vae_model=vqgan_model, **cfg)
        ddpm, _, _ = load_model_checkpoint(ddpm, args.load_checkpoint_ddpm, device)
        ddpm.to(device)
        vae = IntroVAE(**cfg_vae['model'])
        vae, _, _ = load_model_checkpoint(vae, args.vae_path, device)
        vae.to(device)
        global vae_latent_dim
        vae_latent_dim = cfg_vae['model']['latent_dim']
        global latent_dim
        latent_dim = cfg_vqgan['model']['latent_dim']
        block_size = args.block_size
        sample_images_gen(ddpm, vae, block_size, args.image_count, args.gen_image_path, args.image_size, device)


def sample_images_real(data_loader, n_images, real_image_path):
    count = 0
    for x, _ in tqdm(data_loader, desc="sample_real_images"):
        img = tensor_to_image(torch.squeeze(x))
        img.save(f"{real_image_path}/{count}.jpg")

        count += 1
        if count == n_images:
            break

def sample_from_vae(n_images, model, device):
    z = torch.randn(n_images, vae_latent_dim).to(device)
    images = model.decode(z)
    return images
def sample_images_gen(model, vae, block_size, n_images, image_path, image_size, device):
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
        images_decoded = images = [0]*model.n_steps
        channels = 3
        img = torch.randn((n_images, channels, image_size//2, image_size//2), device=device)
        for i in range(len(images)):
            images[i] = img
        img = model.encode(img)
        prev_block = torch.rand_like(img[:, :, :block_size, :block_size]).to(device)
        # prev_block = model.encode(prev_block)
        low_res_cond = sample_from_vae(n_images, vae, device)
        low_res_cond = model.encode(low_res_cond)
        low_res_cond = F.resize(low_res_cond, [block_size], antialias = True)

        position = 0
        for i in range(0, img.shape[-1], block_size):
            for j in range(0, img.shape[-1], block_size):
                # if j==0 and i>0:
                #     prev_block = img[:,:,i-block_size:i, j:j+block_size]
                #     prev_block = model.encode(prev_block)
                block_pos = torch.full((n_images,),position, dtype=torch.int64).to(device)
                curr_block = model.sample(block_size, prev_block, block_pos, low_res_cond, batch_size=n_images, channels=latent_dim)
                curr_block[0] = curr_block[0] - low_res_cond 
                prev_block = curr_block[0]
                position += 1
                for k in range(len(curr_block)):
                    images[k][:, :, i:i+block_size, j:j+block_size] = curr_block[k]
        for k in range(len(images)):
            images_decoded[k] = model.decode(images[k])
        # images = model.sample(16, batch_size=sample_size, channels=latent_dim, sample_step=sample_step)
        images = [img for img in images_decoded[0]]
        images = torch.stack(images)
        # images = model.decode(images)

        for n, img in enumerate(images):
            img = tensor_to_image(img)
            img.save(f"{image_path}/{step_count}_{n}.jpg")

        n_images -= sample_size
        step_count += 1
'''
def validate(model, data_loader, block_size, vae, device):
    model.eval()
    x, _ = next(iter(data_loader))
    x = x.to(device)
    x = model.encode(x)
    n_images = 8
    _, c, w, h = x.size()
    images_decoded = images = [0]*model.n_steps
    img = torch.ones((n_images, c, w, h), device=device)
    for i in range(len(images)):
        images[i] = img
    prev_block = torch.rand_like(img[:, :, :block_size, :block_size]).to(device)
    low_res_cond = sample_from_vae(n_images, vae, device)
    low_res_cond = model.encode(low_res_cond)
    low_res_cond = F.resize(low_res_cond, [block_size], antialias = True)
    position = 0
    for i in range(0, img.shape[-1], block_size):
        for j in range(0, img.shape[-1], block_size):
            block_pos = torch.full((n_images,),position, dtype=torch.int64).to(device)
            curr_block = model.sample(block_size, prev_block, block_pos, low_res_cond, batch_size=n_images, channels=latent_dim)
            curr_block[0] = curr_block[0] - low_res_cond 
            # curr_block[0] = curr_block[0] - prev_block
            prev_block = curr_block[0]
            position += 1
            for k in range(len(curr_block)):
                images[k][:, :, i:i+block_size, j:j+block_size] = curr_block[k]
    for k in range(len(images)):
        images_decoded[k] = model.decode(images[k])
    logger.tensorboard.add_figure('Val: DDPM',
                                  get_sample_images_for_ddpm(images, n_ims=n_images),
                                  global_step=logger.global_train_step)

'''

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        raise e
