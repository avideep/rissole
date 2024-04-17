import argparse
import os
import pathlib
import time
from datetime import datetime
import numpy as np
import torch
import yaml

from tqdm import tqdm
import torchvision.transforms.functional as F
from dataloader import PlantNet, CIFAR10, CelebA, CelebAHQ, ImageNet100
from dsetbuilder_clip import DSetBuilder
from model import VQGANLight, VAE, IntroVAE
from model.ddpm.ddpm import DDPM
from model.unet import UNet
from model.unet.unet_light import UNetLight
from utils.helpers import timer, save_model_checkpoint, load_model_checkpoint, log2tensorboard_ddpm
from utils.logger import Logger
from utils.helpers import count_parameters
from utils.visualization import get_sample_images_for_ddpm

# TODO: check if this is necessary
# from: https://stackoverflow.com/questions/20554074/sklearn-omp-error-15-initializing-libiomp5md-dll-but-found-mk2iomp5md-dll-a
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

CHECKPOINT_DIR = os.path.join(pathlib.Path(__file__).parent.resolve(), 'checkpoints')
LOG_DIR = os.path.join(pathlib.Path(__file__).parent.resolve(), 'logs')
TIMESTAMP = datetime.now().strftime('%y-%m-%d_%H%M%S')

parser = argparse.ArgumentParser(description="PyTorch Second Stage Training")
parser.add_argument('--name', '-n', default='',
                    type=str, metavar='NAME', help='Model name and folder where logs are stored')
parser.add_argument('--epochs', default=200,
                    type=int, metavar='N', help='Number of epochs to run (default: 100)')
parser.add_argument('--batch-size', default=16, metavar='N',
                    type=int, help='Mini-batch size (default: 64)')
parser.add_argument('--dset-batch-size', default=32, metavar='N',
                    type=int, help='Mini-batch size (default: 32)')
parser.add_argument('--image-size', default=224, metavar='N',
                    type=int, help='Size that images should be resized to before processing (default: 128)')
parser.add_argument('--block-size', default=14, metavar='N',
                    type=int, help='Size of the block that the image will be divided by.')
parser.add_argument('--k', default=10, metavar='N',
                    type=int, help='Number of nearest neighbors to search.')
parser.add_argument('--image-channels', default=3, metavar='N',
                    type=int, help='Number of image channels (default: 3)')
parser.add_argument('--num-workers', default=0, metavar='N',
                    type=int, help='Number of workers for the dataloader (default: 0)')
parser.add_argument('--lr', default=2e-4,
                    type=float, metavar='LR', help='Initial learning rate (default: 0.0002)')
parser.add_argument('--config', default='configs/ddpm_linear.yaml',
                    metavar='PATH', help='Path to model config file (default: configs/ddpm_linear.yaml)')
parser.add_argument('--unet-config', default='configs/unet.yaml',
                    metavar='PATH', help='Path to unet model config file (default: configs/unet.yaml)')
parser.add_argument('--data-config', default='configs/data_se.yaml',
                    metavar='PATH', help='Path to model config file (default: configs/data_se.yaml)')
parser.add_argument('--data', '-d', default='CelebA',
                        type=str, metavar='data', help='Dataset Name. Please enter CelebA, CelebAHQ, or CIFAR10. Default: CelebA')
parser.add_argument('--gpus', default=0, type=int,
                    nargs='+', metavar='GPUS', help='If GPU(s) available, which GPU(s) to use for training.')
parser.add_argument('--ckpt-save', default=True, action='store_true',
                    dest='save_checkpoint', help='Save checkpoints to folder')
parser.add_argument('--load-ddpm', default=None, metavar='PATH',
                    help='Load model checkpoint and continue training')
parser.add_argument('--load-ckpt_ddpm', default=None, metavar='PATH',
                    dest='load_checkpoint_ddpm', help='Load model checkpoint and continue training')
parser.add_argument('--load-ckpt_unet', default=None, metavar='PATH',
                    dest='load_checkpoint_unet', help='Load model checkpoint and continue training')
parser.add_argument('--vqgan-path', default='checkpoints/vqgan/24-03-18_151152/best_model.pt',
                    metavar='PATH', help='Path to encoder/decoder model checkpoint (default: empty)')
parser.add_argument('--vqgan-config', default='configs/vqgan_cifar10.yaml',
                    metavar='PATH', help='Path to model config file (default: configs/vqgan.yaml)')
parser.add_argument('--vae-path', default='checkpoints/vae/24-02-15_130409/best_model.pt',
                    metavar='PATH', help='Path to encoder/decoder model checkpoint (default: empty)')
parser.add_argument('--vae-config', default='configs/vae.yaml',
                    metavar='PATH', help='Path to model config file (default: configs/vaeyaml)')
parser.add_argument('--use-low-res', action='store_true',
                    help='Whether to condition the model with a low resolution whole image sampled from a VAE')
parser.add_argument('--use-prev-block', action='store_true',
                    help='Whether to condition the model with the previous block')
parser.add_argument('--use-cfg', action='store_true',
                    help='Whether to use classifier-free guidance')
parser.add_argument('--guidance-probability', default=0.7, type=float,
                    help='probability of unconditional generation (default: 0.8)')
parser.add_argument('--guidance-weight', default=10, type=int,
                    help='weight on unconditional generaton. (default: 5)')


logger = Logger(LOG_DIR)
latent_dim = None


def main():
    args = parser.parse_args()
    for name, val in vars(args).items():
        print("{:<16}: {}".format(name, val))

    # setup paths and logging
    args.name = 'second_stage/' + os.path.splitext(os.path.basename(args.config))[0]
    running_log_dir = os.path.join(LOG_DIR, args.name, f'{TIMESTAMP}')
    running_ckpt_dir_ddpm = os.path.join(CHECKPOINT_DIR, args.name, f'{TIMESTAMP}', "ddpm")
    running_ckpt_dir_unet = os.path.join(CHECKPOINT_DIR, args.name, f'{TIMESTAMP}', "unet")
    print("{:<16}: {}".format('logdir', running_log_dir))
    print("{:<16}: {}".format('ckpt_dir_ddpm', running_ckpt_dir_ddpm))
    print("{:<16}: {}".format('ckpt_dir_unet', running_ckpt_dir_unet))

    if args.save_checkpoint and not os.path.exists(running_ckpt_dir_ddpm):
        os.makedirs(running_ckpt_dir_ddpm)

    if args.save_checkpoint and not os.path.exists(running_ckpt_dir_unet):
        os.makedirs(running_ckpt_dir_unet)

    global logger
    logger = Logger(running_log_dir, tensorboard=True)

    # setup GPU
    args.gpus = args.gpus if isinstance(args.gpus, list) else [args.gpus]
    if len(args.gpus) == 1:
        device = torch.device(f'cuda:{args.gpus[0]}' if torch.cuda.is_available() else 'cpu')
    else:
        raise ValueError('Currently multi-gpu training is not possible')
    # load data
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
    # read config file for model
    cfg = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)
    cfg_unet = yaml.load(open(args.unet_config, 'r'), Loader=yaml.Loader)
    cfg_vqgan = yaml.load(open(args.vqgan_config, 'r'), Loader=yaml.Loader)
    cfg_vae = yaml.load(open(args.vae_config,'r'),Loader=yaml.Loader)
    vae = None
    if args.use_low_res:
        vae = VAE(**cfg_vae['model'])
        # vae = IntroVAE(**cfg_vae['model'])
        vae, _, _ = load_model_checkpoint(vae, args.vae_path, device)
        vae.to(device)
        global vae_latent_dim
        vae_latent_dim = cfg_vae['model']['latent_dim']
        cfg_unet['cond_emb_dim'] = 2 * cfg_unet['cond_emb_dim']
    vqgan_model = VQGANLight(**cfg_vqgan['model'])
    vqgan_model, _, _ = load_model_checkpoint(vqgan_model, args.vqgan_path, device)
    vqgan_model.to(device)
    global latent_dim
    latent_dim = cfg_vqgan['model']['latent_dim']
    cfg_unet['in_channels'] = (args.k + 1) * latent_dim

    unet = UNetLight(**cfg_unet)
    unet.to(device)

    ddpm = DDPM(eps_model=unet, vae_model=vqgan_model, **cfg)
    ddpm.to(device)

    dset = DSetBuilder(data, args.k)

    print("{:<16}: {}".format('DDPM model params', count_parameters(ddpm)))

    block_size = args.block_size
    optimizer = torch.optim.Adam(unet.parameters(), args.lr)

    # resume training
    if args.load_checkpoint_ddpm:
        unet, start_epoch, global_train_step = load_model_checkpoint(unet, args.load_checkpoint_unet, device)
        ddpm, start_epoch, global_train_step = load_model_checkpoint(ddpm, args.load_checkpoint_ddpm, device)
        logger.global_train_step = global_train_step
        args.epochs += start_epoch
    else:
        start_epoch = 0

    # start run
    logger.log_hparams({**cfg, **vars(args)})
    t_start = time.time()
    prev_loss = torch.inf
    for epoch in range(start_epoch, args.epochs):

        logger.init_epoch(epoch)
        logger.global_train_step = logger.running_epoch
        print(f"Epoch [{epoch + 1} / {args.epochs}]")

        train(ddpm, data, dset, optimizer, device, args)

        validate(ddpm, dset, device, args)

        # logging
        output = ' - '.join([f'{k}: {v.avg:.4f}' for k, v in logger.epoch.items()])
        print(output)
        loss = logger.epoch['ema_loss'].avg
        if loss < prev_loss:
        # save logs and checkpoint
            logger.save()
            if args.save_checkpoint:
                save_model_checkpoint(unet, f"{running_ckpt_dir_unet}", logger)
                save_model_checkpoint(ddpm, f"{running_ckpt_dir_ddpm}", logger)
            prev_loss = loss

        log2tensorboard_ddpm(logger, 'Train', ['ema_loss', 'loss'])

    elapsed_time = timer(t_start, time.time())
    print(f"Total training time: {elapsed_time}")

def debug(model,data_loader,device):
    x, _ = next(iter(data_loader))
    x = x.to(device)
    print(model.encode(x).shape)


def train(model, data, dset, optimizer, device, args):
    model.train()

    ema_loss = None
    p = args.guidance_probability
    for x, _ in tqdm(data.train, desc="Training"):
        neighbor_ids = dset.get_neighbor_ids(x)
        x = x.to(device)
        x = model.encode(x)
        optimizer.zero_grad()
        neighbors = dset.get_neighbors(neighbor_ids, x.size()).to(device)
        print(x.shape)
        print(neighbors.shape)
        loss = model.p_losses2(x, neighbors)
        loss.backward()
        optimizer.step()
        loss = loss.item()
        if ema_loss is None:
            ema_loss = loss
        else:
            ema_loss = 0.9 * ema_loss + 0.1 * loss

        metrics = {'ema_loss': ema_loss, 'loss': loss}
        logger.log_metrics(metrics, phase='Train', aggregate=True, n=x.shape[0])

@torch.no_grad()
def validate(model, dset, device, args):
    model.eval()
    n_images = 8
    x_query = dset.get_rand_queries(n_images)
    neighbor_ids = dset.get_neighbor_ids(x_query)
    neighbors = dset.get_neighbors(neighbor_ids, n_images, latent_dim).to(device)
    images = model.sample(args.img_size, neighbors, batch_size=n_images, channels=latent_dim)
    images = [model.decode(img) for img in images]

    logger.tensorboard.add_figure('Val: DDPM',
                                  get_sample_images_for_ddpm(images, n_ims=n_images),
                                  global_step=logger.global_train_step)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.save()
        raise e
