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
from dataloader import PlantNet, CIFAR10, CelebA, CelebAHQ, DSetBuilder
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
parser.add_argument('--image-size', default=128, metavar='N',
                    type=int, help='Size that images should be resized to before processing (default: 128)')
parser.add_argument('--block-size', default=32, metavar='N',
                    type=int, help='Size of the block that the image will be divided by.')
parser.add_argument('--k', default=10, metavar='N',
                    type=int, help='Size of the block that the image will be divided by.')
parser.add_argument('--image-channels', default=3, metavar='N',
                    type=int, help='Number of image channels (default: 3)')
parser.add_argument('--num-workers', default=0, metavar='N',
                    type=int, help='Number of workers for the dataloader (default: 0)')
parser.add_argument('--lr', default=0.00002,
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
parser.add_argument('--vqgan-path', default='checkpoints/vqgan/24-02-15_130652/best_model.pt',
                    metavar='PATH', help='Path to encoder/decoder model checkpoint (default: empty)')
parser.add_argument('--vqgan-config', default='configs/vqgan_cifar10.yaml',
                    metavar='PATH', help='Path to model config file (default: configs/vqgan.yaml)')
parser.add_argument('--vae-path', default='checkpoints/vae/24-02-15_130409/best_model.pt',
                    metavar='PATH', help='Path to encoder/decoder model checkpoint (default: empty)')
parser.add_argument('--vae-config', default='configs/vae.yaml',
                    metavar='PATH', help='Path to model config file (default: configs/vaeyaml)')
parser.add_argument('--use-low-res', action='store_true',
                    help='Whether to condition the model with a low resolution whole image sampled from a VAE')
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
    if args.debug:
        data = CIFAR10(args.batch_size)
        # data = CelebA(args.batch_size)
    else:
        data = CelebAHQ(args.batch_size, device=device)
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

    unet = UNetLight(**cfg_unet)
    unet.to(device)

    ddpm = DDPM(eps_model=unet, vae_model=vqgan_model, **cfg)
    ddpm.to(device)
    
    dset = DSetBuilder(data, 10, vqgan_model)

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

        train(ddpm, data, dset, optimizer, block_size, vae, device, args)

        validate(ddpm, data, dset, block_size, vae, device, args)

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


def train(model, data, dset, optimizer, block_size, vae, device, args):
    model.train()

    ema_loss = None
    p = args.guidance_probability
    for x, _ in tqdm(data.train, desc="Training"):
        x = x.to(device)
        x = model.encode(x)
        if args.use_cfg:
            if args.use_low_res  and  np.random.choice([1, 0], p=[1-p, p]): # setting the condition to None as per the guidance probability, should the condition be used
                x_hat = sample_from_vae(x.shape[0],vae, device)
                x_resized = model.encode(x_hat)
                low_res_cond = F.resize(x_resized, [block_size], antialias = True)
            else:
                low_res_cond = None
        elif args.use_low_res: # if cfg is not used but low res cond is going to be used
            x_hat = sample_from_vae(x.shape[0],vae, device)
            x_resized = model.encode(x_hat)
            low_res_cond = F.resize(x_resized, [block_size], antialias = True)
        else: # if nothing is used
            low_res_cond = None
        prev_block = torch.rand_like(x[:, :, :block_size, :block_size]).to(device)
        optimizer.zero_grad()
        position = 0
        loss_agg = 0
        neighbor_ids = dset.get_neighbor_ids(model.decode(prev_block))
        for i in range(0, x.shape[-1], block_size):
            for j in range(0, x.shape[-1], block_size):
                if j==0 and i>0:
                        prev_block = x[:,:,i-block_size:i, j:j+block_size]
                block_pos = torch.full((x.size(0),),position, dtype=torch.int64).to(device)
                curr_block = x[:, :, i:i+block_size, j:j+block_size]
                # print(prev_block.shape)
                neighbors = dset.get_neighbors(neighbor_ids, position, block_size, args.batch_size).to(device)
                loss = model.p_losses2(curr_block, neighbors, position = block_pos, low_res_cond = low_res_cond)
                prev_block = curr_block
                loss_agg += loss
                position += 1
        loss_agg.backward()
        optimizer.step()
        loss_agg = loss_agg.item()/position #average loss over all the blocks
        if ema_loss is None:
            ema_loss = loss_agg
        else:
            ema_loss = 0.9 * ema_loss + 0.1 * loss_agg

        metrics = {'ema_loss': ema_loss, 'loss': loss_agg}
        logger.log_metrics(metrics, phase='Train', aggregate=True, n=curr_block.shape[0])

@torch.no_grad()
def sample_from_vae(n_images, model, device):
    z = torch.randn(n_images, vae_latent_dim).to(device)
    images = model.decode(z)
    return images

@torch.no_grad()
def validate(model, data, dset, block_size, vae, device, args):
    model.eval()
    x, _ = next(iter(data.val))
    x = x.to(device)
    x = model.encode(x)
    n_images = 8
    _, c, w, h = x.size()
    images_decoded = images = [0]*model.n_steps
    img = torch.ones((n_images, c, w, h), device=device)
    for i in range(len(images)):
        images[i] = img
    prev_block = torch.rand_like(img[:, :, :block_size, :block_size]).to(device)
    if args.use_low_res: 
        low_res_cond = sample_from_vae(n_images, vae, device)
        low_res_cond = model.encode(low_res_cond)
        low_res_cond = F.resize(low_res_cond, [block_size], antialias = True)
    else:
        low_res_cond = None
    position = 0
    neighbor_ids = dset.get_neighbor_ids(model.decode(prev_block))
    w = args.guidance_weight
    for i in range(0, img.shape[-1], block_size):
        for j in range(0, img.shape[-1], block_size):
            if j==0 and i>0:
                prev_block = curr_block[0][:,:,i-block_size:i, j:j+block_size]
            block_pos = torch.full((n_images,),position, dtype=torch.int64).to(device)
            neighbors = dset.get_neighbors(neighbor_ids, position, block_size, n_images).to(device)
            if args.use_low_res and args.use_cfg:
                curr_block_uncond = model.sample(block_size, prev_block, block_pos, low_res_cond = None, batch_size=n_images, channels=latent_dim) #sampling strategy for classifier-free guidance (CFG)
                curr_block_cond = model.sample(block_size, prev_block, block_pos, low_res_cond, batch_size=n_images, channels=latent_dim) #sampling strategy for classifier-free guidance 
                curr_block = [(1 + w)*curr_block_cond[i] - w*curr_block_uncond[i] for i in range(model.n_steps)] #sampling strategy for classifier-free guidance 
            elif args.use_low_res:
                curr_block = model.sample(block_size, prev_block, block_pos, low_res_cond, batch_size=n_images, channels=latent_dim) # if CFG is not used 
            else:
                curr_block = model.sample(block_size, neighbors, block_pos, low_res_cond = None, batch_size=n_images, channels=latent_dim) # if CFG is not used and low-res-conditioning is also not used

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
