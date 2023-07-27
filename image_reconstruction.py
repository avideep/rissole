import os
import yaml
import torch
import argparse
from PIL import Image

from model import VQGAN, VQGANLight, VQVAE
from utils.helpers import load_model_checkpoint
from utils.visualization import tensor_to_image
from dataloader import PlantNet, CelebA


parser = argparse.ArgumentParser(description="Reconstruct images.")
parser.add_argument('--dst', '-d', default='',
                    type=str, metavar='PATH', help='Target folder.')
parser.add_argument('--model', '-m', default='vqgan', choices=['vqgan', 'vqvae'],
                    type=str, metavar='NAME', help='Which model to use.')
parser.add_argument('--config', default='configs/vqgan_cifar10.yaml',
                    metavar='PATH', help='Path to model config file (default: configs/vqgan.yaml)')
parser.add_argument('--ckpt', default='checkpoints/vqgan/23-07-25_093150/best_model.pt', metavar='PATH',
                    dest='ckpt', help='Load model checkpoint.')
parser.add_argument('--block-size', default=32, metavar='N',
                    type=int, help='Size of the block that the image will be divided by.')
parser.add_argument('--prefix', default='',
                    type=str, metavar='PREFIX', help='Prefix for image naming.')
parser.add_argument('-n', default=4, metavar='N',
                    type=int, help='Number of reconstructed images.')
parser.add_argument('--data-config', default=None, metavar='PATH',
                    help='Path to model config file (default: None)')
parser.add_argument('--gpus', default=None, nargs='+', metavar='GPUS',
                    help='If GPU(s) available, which GPU(s) to use for training.')
parser.add_argument('--save-original', action='store_true',
                    dest='save_original', help='Whether or not to save original images.')


IMG_DIR = 'data/reconstructions'


BASE_CFG = {
    'data_dir': '/home/johannes-f/Documents/datasets/data_128',
    'is_preprocessed': True
}


def main():
    args = parser.parse_args()

    args.gpus = args.gpus if isinstance(args.gpus, list) else [args.gpus]
    if len(args.gpus) == 1:
        device = torch.device(f'cuda:{args.gpus[0]}' if torch.cuda.is_available() and args.gpus[0] is not None else 'cpu')
    else:
        raise ValueError('Currently multi-gpu training is not possible')
    print("{:<16}: {}".format('device', device))

    img_dir = os.path.join(IMG_DIR, args.dst)
    if not os.path.exists(img_dir):
        os.makedirs(img_dir)

    # load data
    if args.data_config is None:
        data = CelebA(args.batch_size)
    else:
        data_cfg = yaml.load(open(args.data_config, 'r'), Loader=yaml.Loader)
        data = PlantNet(**data_cfg, batch_size=args.n)

    # load model
    cfg = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)
    if args.model.lower() == 'vqgan':
        model = VQGANLight(**cfg['model'])
    elif args.model.lower() == 'vqvae':
        model = VQVAE(**cfg['model'])
    else:
        raise Exception(f'"{args.model}" is no valid model!')
    model.to(device)

    model, _, _ = load_model_checkpoint(model, args.ckpt, device)
    block_size = args.block_size
    # get images
    x, _ = iter(data.val).next()
    x = x.to(device)
    x_recon = torch.randn_like(x)
    # reconstruct images
    print("Reconstruct images...")
    model.eval()
    with torch.no_grad():
        for i in range(0, x.shape[-1], block_size):
            for j in range(0, x.shape[-1], block_size):
                block = x[:, :, i:i+block_size, j:j+block_size]
                x_hat, _, _ = model(block)
                x_recon[:, :, i:i+block_size, j:j+block_size] = x_hat
        print("shape:", x_recon.shape)

    ims = [tensor_to_image(t) for t in x]
    recs = [tensor_to_image(t) for t in x_recon]

    # save images
    prefix = args.prefix + '_' if args.prefix != "" else ''
    for i, (im, rec) in enumerate(zip(ims, recs)):
        rec.save(os.path.join(img_dir, f'{args.model}_{prefix}{i}_recon.png'))
        if args.save_original:
            im.save(os.path.join(img_dir, f'{args.model}_{prefix}{i}_original.png'))


if __name__ == "__main__":
    main()
