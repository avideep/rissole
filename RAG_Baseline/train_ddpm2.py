import argparse
import os
import pathlib
import time
from datetime import datetime

import torch
import yaml

from tqdm import tqdm
import torchvision.transforms.functional as F
from dataloader import PlantNet, CIFAR10, CelebA
from model import VQGANLight, VAE
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
parser.add_argument('--epochs', default=100,
                    type=int, metavar='N', help='Number of epochs to run (default: 100)')
parser.add_argument('--batch-size', default=128, metavar='N',
                    type=int, help='Mini-batch size (default: 64)')
parser.add_argument('--image-size', default=64, metavar='N',
                    type=int, help='Size that images should be resized to before processing (default: 128)')
parser.add_argument('--block-size', default=16, metavar='N',
                    type=int, help='Size of the block that the image will be divided by.')
parser.add_argument('--image-channels', default=3, metavar='N',
                    type=int, help='Number of image channels (default: 3)')
parser.add_argument('--num-workers', default=0, metavar='N',
                    type=int, help='Number of workers for the dataloader (default: 0)')
parser.add_argument('--lr', default=0.0001,
                    type=float, metavar='LR', help='Initial learning rate (default: 0.0002)')
parser.add_argument('--config', default='configs/ddpm_linear.yaml',
                    metavar='PATH', help='Path to model config file (default: configs/ddpm_linear.yaml)')
parser.add_argument('--unet-config', default='configs/unet.yaml',
                    metavar='PATH', help='Path to unet model config file (default: configs/unet.yaml)')
parser.add_argument('--data-config', default='configs/data_se.yaml',
                    metavar='PATH', help='Path to model config file (default: configs/data_se.yaml)')
parser.add_argument('--debug', action='store_true',
                    help='If true, trains on CIFAR10')
parser.add_argument('--gpus', default=0, type=int,
                    nargs='+', metavar='GPUS', help='If GPU(s) available, which GPU(s) to use for training.')
parser.add_argument('--ckpt-save', default=True, action='store_true',
                    dest='save_checkpoint', help='Save checkpoints to folder')
parser.add_argument('--load-ckpt_ddpm', default=None, metavar='PATH',
                    dest='load_checkpoint_ddpm', help='Load model checkpoint and continue training')
parser.add_argument('--load-ckpt_unet', default=None, metavar='PATH',
                    dest='load_checkpoint_unet', help='Load model checkpoint and continue training')
parser.add_argument('--log-save-interval', default=5, type=int, metavar='N',
                    dest='save_interval', help="Interval in which logs are saved to disk (default: 5)")
parser.add_argument('--vqgan-path', default='checkpoints/vqgan/23-12-18_175014/best_model.pt',
                    metavar='PATH', help='Path to encoder/decoder model checkpoint (default: empty)')
parser.add_argument('--vqgan-config', default='configs/vqgan_cifar10.yaml',
                    metavar='PATH', help='Path to model config file (default: configs/vqgan.yaml)')


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
    if args.debug:
        #data = CIFAR10(args.batch_size)
        data = CelebA(args.batch_size)
    else:
        data_cfg = yaml.load(open(args.data_config, 'r'), Loader=yaml.Loader)
        data = PlantNet(**data_cfg, batch_size=args.batch_size, image_size=args.image_size,
                        num_workers=args.num_workers)

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

    unet = UNetLight(**cfg_unet)
    unet.to(device)

    ddpm = DDPM(eps_model=unet, vae_model=vqgan_model, **cfg)
    print("{:<16}: {}".format('DDPM model params', count_parameters(ddpm)))
    ddpm.to(device)

    # vae = VAE(**cfg_vae['model'])
    # vae, _, _ = load_model_checkpoint(vae, args.vae_path, device)
    # vae.to(device)
    # global vae_latent_dim
    # vae_latent_dim = cfg_vae['model']['latent_dim']

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
    # debug(ddpm, data.train, device)

    # start run
    logger.log_hparams({**cfg, **vars(args)})
    t_start = time.time()
    prev_loss = torch.inf
    for epoch in range(start_epoch, args.epochs):

        logger.init_epoch(epoch)
        logger.global_train_step = logger.running_epoch
        print(f"Epoch [{epoch + 1} / {args.epochs}]")

        train(ddpm, data.train, optimizer, block_size, device)

        validate(ddpm, data.val, block_size, device)

        # logging
        output = ' - '.join([f'{k}: {v.avg:.4f}' for k, v in logger.epoch.items()])
        print(output)
        loss = logger.epoch['ema_loss'].avg
        if loss < prev_loss:
        # save logs and checkpoint
        #if (epoch + 1) % args.save_interval == 0 or (epoch + 1) == args.epochs:
            logger.save()
            if args.save_checkpoint:
                save_model_checkpoint(unet, f"{running_ckpt_dir_unet}", logger)
                save_model_checkpoint(ddpm, f"{running_ckpt_dir_ddpm}", logger)
            prev_loss = loss

        log2tensorboard_ddpm(logger, 'Train', ['ema_loss', 'loss'])

    elapsed_time = timer(t_start, time.time())
    print(f"Total training time: {elapsed_time}")


# def train(model, train_loader, optimizer, device):
#     model.train()

#     ema_loss = None
#     for x, _ in tqdm(train_loader, desc="Training"):
#         optimizer.zero_grad()
#         x = x.to(device)
#         loss = model.p_losses(x)
#         loss.backward()
#         optimizer.step()

#         if ema_loss is None:
#             ema_loss = loss.item()
#         else:
#             ema_loss = 0.9 * ema_loss + 0.1 * loss.item()

#         metrics = {'ema_loss': ema_loss, 'loss': loss}
#         logger.log_metrics(metrics, phase='Train', aggregate=True, n=x.shape[0])
def debug(model,data_loader,device):
    x, _ = next(iter(data_loader))
    x = x.to(device)
    print(model.encode(x).shape)


def train(model, train_loader, optimizer, block_size, device):
    model.train()

    ema_loss = None
    for x, _ in tqdm(train_loader, desc="Training"):
        x = x.to(device)
        x = model.encode(x) # encoding the whole image
        # x_resized = F.resize(x, [block_size], antialias = True)
        # x_resized = sample_from_vae(x.shape[0],vae, device)
        prev_block = torch.rand_like(x[:, :, :block_size, :block_size]).to(device)
        optimizer.zero_grad()
        loss_agg = 0
        position = 0
        for i in range(0, x.shape[-1], block_size): # dividing the encoded representation of the image into blocks
            for j in range(0, x.shape[-1], block_size):
                # if j==0 and i>0:
                #         prev_block = x[:,:,i-block_size:i, j:j+block_size]
                block_pos = torch.full((x.size(0),),position, dtype=torch.int64).to(device)
                curr_block = x[:, :, i:i+block_size, j:j+block_size]
                loss = model.p_losses2(curr_block, prev_block, block_pos)
                prev_block = curr_block
                loss_agg += loss.item()
                loss.backward()
                position += 1
        optimizer.step()

        if ema_loss is None:
            ema_loss = loss_agg
        else:
            ema_loss = 0.9 * ema_loss + 0.1 * loss_agg

        metrics = {'ema_loss': ema_loss, 'loss': loss_agg}
        logger.log_metrics(metrics, phase='Train', aggregate=True, n=curr_block.shape[0])
    # print("#"*30)
    # print(model.count)
    # print("#"*30)
    # model.count = 0

# @torch.no_grad()
# def sample_from_vae(n_images, model, device):
#     z = torch.randn(n_images, vae_latent_dim).to(device)
#     images = model.decode(z)
#     return images

@torch.no_grad()
def validate(model, data_loader, block_size, device):
    model.eval()
    x, _ = next(iter(data_loader))
    x = x.to(device)
    x = model.encode(x)
    n_images = 8
    _, c, w, h = x.size()
    images_decoded = images = [0]*model.n_steps
    img_decoded = img = torch.ones((n_images, c, w, h), device=device)
    for i in range(len(images)):
        images[i] = img
        images_decoded[i] = img_decoded
    prev_block = torch.rand_like(img[:, :, :block_size, :block_size]).to(device)
    # prev_block = model.encode(prev_block)
    # low_res_cond = sample_from_vae(n_images, vae, device)
    # low_res_cond = model.encode(low_res_cond)
    position = 0
    for i in range(0, img.shape[-1], block_size):
        for j in range(0, img.shape[-1], block_size):
            # if j==0 and i>0:
            #     prev_block = img[:,:,i-block_size:i, j:j+block_size]
            #     prev_block = model.encode(prev_block)
            block_pos = torch.full((n_images,),position, dtype=torch.int64).to(device)
            curr_block = model.sample(16, prev_block, block_pos, batch_size=n_images, channels=latent_dim)
            prev_block = curr_block[0]
            position += 1
            for k in range(len(curr_block)):
                images[k][:, :, i:i+block_size, j:j+block_size] = curr_block[k]
    for k in range(len(images)):
        images_decoded[k] = model.decode(images[k])
    logger.tensorboard.add_figure('Val: DDPM',
                                  get_sample_images_for_ddpm(images_decoded, n_ims=n_images),
                                  global_step=logger.global_train_step)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.save()
        raise e
