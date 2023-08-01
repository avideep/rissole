import argparse
import os

import torch
import yaml

from tqdm import tqdm

from utils.visualization import tensor_to_image
from dataloader import CelebA
from model import VQGANLight, VAE
from utils.helpers import load_model_checkpoint

# from: https://stackoverflow.com/questions/20554074/sklearn-omp-error-15-initializing-libiomp5md-dll-but-found-mk2iomp5md-dll-a
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

parser = argparse.ArgumentParser(description="PyTorch Second Stage Training")
parser.add_argument('--image-size', default=32, metavar='N',
                    type=int, help='Size that images should be resized to before processing (default: 128)')
parser.add_argument('--image-count', default=50,
                    type=int, help='number of images that should be generated for comparison')
parser.add_argument('--vae-path', default='checkpoints/vae/23-08-01_105355/best_model.pt',
                    metavar='PATH', help='Path to encoder/decoder model checkpoint (default: empty)')
parser.add_argument('--vae-config', default='configs/vae.yaml',
                    metavar='PATH', help='Path to model config file (default: configs/vqgan.yaml)')
parser.add_argument('--real-image-path', default='real_images',
                    metavar='PATH', help='Path to load samples from the source images into')
parser.add_argument('--gen-image-path', default='samples_vae',
                    metavar='PATH', help='Path to generated images')
parser.add_argument('--gpus', default=0, type=int,
                    nargs='+', metavar='GPUS', help='If GPU(s) available, which GPU(s) to use for training.')
parser.add_argument('--sample_real', action='store_true',
                    help='If true, samples images from the ground truth imageset')
parser.add_argument('--sample_gen', action='store_true',
                    help='If true, samples images from the ddpm')
parser.add_argument('--data-config', default='configs/data_se.yaml',
                    metavar='PATH', help='Path to model config file (default: configs/data_se.yaml)')


def main():
    args = parser.parse_args()
    for name, val in vars(args).items():
        print("{:<16}: {}".format(name, val))

    if args.sample_real:
        if args.real_image_path and not os.path.exists(args.real_image_path):
            os.makedirs(args.real_image_path)
        data = CelebA(batch_size=1, image_size=args.image_size)
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
        cfg_vae = yaml.load(open(args.vae_config, 'r'), Loader=yaml.Loader)

        vae_model = VAE(**cfg_vae['model'])
        vae_model, _, _ = load_model_checkpoint(vae_model, args.vae_path, device)
        vae_model.to(device)

        global latent_dim
        latent_dim = cfg_vae['model']['latent_dim']
        sample_images_gen(vae_model, args.image_count, args.gen_image_path, device)


def sample_images_real(data_loader, n_images, real_image_path):
    count = 0
    for x, _ in tqdm(data_loader, desc="sample_real_images"):
        img = tensor_to_image(torch.squeeze(x))
        img.save(f"{real_image_path}/{count}.jpg")

        count += 1
        if count == n_images:
            break


def sample_images_gen(model, n_images, image_path, device):
    model.eval()
    max_sample_size = 128
    step_count = 0

    while n_images > 0:
        if n_images >= max_sample_size:
            sample_size = max_sample_size
        else:
            sample_size = n_images
        z = torch.randn(n_images, model.latent_size).to(device)
        images = model.decode(z).cpu()
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
