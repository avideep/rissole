import os
import pathlib
import sys
import time
import yaml
import argparse
from tqdm import tqdm
from datetime import datetime
import torch

from model import VQGAN, VQGANLight
from model.losses import LossFn
from utils.logger import Logger
from utils.helpers import timer
from utils.helpers import load_model_checkpoint, save_model_checkpoint
from utils.helpers import log2tensorboard_vqvae
from utils.helpers import count_parameters
from utils.visualization import get_original_reconstruction_image
from dataloader import CIFAR10, CelebA, CelebAHQ, ImageNet100

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


CHECKPOINT_DIR = os.path.join(pathlib.Path(__file__).parent.resolve(), 'checkpoints')
LOG_DIR = os.path.join(pathlib.Path(__file__).parent.resolve(), 'logs')
TIMESTAMP = datetime.now().strftime('%y-%m-%d_%H%M%S')


parser = argparse.ArgumentParser(description="PyTorch First Stage Training")
parser.add_argument('--name', '-n', default='',
                    type=str, metavar='NAME', help='Model name and folder where logs are stored')
parser.add_argument('--epochs', default=300,
                    type=int, metavar='N', help='Number of epochs to run (default: 2)')
parser.add_argument('--batch-size', default=512, metavar='N',
                    type=int, help='Mini-batch size (default: 64)')
parser.add_argument('--dset-batch-size', default=32, metavar='N',
                    type=int, help='Mini-batch size (default: 32)')
parser.add_argument('--image-size', default=32, metavar='N',
                    type=int, help='Size that images should be resized to before processing (default: 128)')
parser.add_argument('--block-size', default=32, metavar='N',
                    type=int, help='Size of the block that the image will be divided by.')
parser.add_argument('--num-workers', default=0, metavar='N',
                    type=int, help='Number of workers for the dataloader (default: 0)')
parser.add_argument('--lr', default=0.0002,
                    type=float, metavar='LR', help='Initial learning rate (default: 0.0001)')
parser.add_argument('--config', default='configs/vqgan_cifar10.yaml',
                    metavar='PATH', help='Path to model config file (default: configs/vqgan.yaml)')
parser.add_argument('--data-config', default='configs/data_se.yaml',
                    metavar='PATH', help='Path to model config file (default: configs/data_se.yaml)')
parser.add_argument('--data', '-d', default='CIFAR10',
                        type=str, metavar='data', help='Dataset Name. Please enter CelebA, CelebAHQ, or CIFAR10. Default: CelebA')
parser.add_argument('--gpus', default=0, type=int,
                    nargs='+', metavar='GPUS', help='If GPU(s) available, which GPU(s) to use for training.')
parser.add_argument('--ckpt-save', default=True, action='store_true',
                    dest='save_checkpoint', help='Save checkpoints to folder')
parser.add_argument('--load-ckpt', default=None, metavar='PATH',
                    dest='load_checkpoint', help='Load model checkpoint and continue training')
parser.add_argument('--log-save-interval', default=5, type=int, metavar='N',
                    dest='save_interval', help="Interval in which logs are saved to disk (default: 5)")


logger = Logger(LOG_DIR)


def main():
    args = parser.parse_args()
    for name, val in vars(args).items():
        print("{:<16}: {}".format(name, val))

    # setup paths and logging
    args.name = 'vqgan/' + args.name
    running_log_dir = os.path.join(LOG_DIR, args.name, f'{TIMESTAMP}')
    running_ckpt_dir = os.path.join(CHECKPOINT_DIR, args.name, f'{TIMESTAMP}')
    print("{:<16}: {}".format('logdir', running_log_dir))
    print("{:<16}: {}".format('ckpt_dir', running_ckpt_dir))

    if args.save_checkpoint and not os.path.exists(running_ckpt_dir):
        os.makedirs(running_ckpt_dir)

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
        data = CelebA(args.batch_size)
    elif args.data == 'CIFAR10':
        data = CIFAR10(args.batch_size)
    elif args.data == 'ImageNet100':
        data = ImageNet100(args.batch_size, dset_batch_size= args.dset_batch_size)
    else:
        data = CelebAHQ(args.batch_size, dset_batch_size= args.dset_batch_size, device=device)
    # read config file for model
    cfg = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)

    # create model and optimizer
    model = VQGANLight(**cfg['model'])
    #model = VQGAN(**cfg['model'])
    print("{:<16}: {}".format('model params', count_parameters(model)))
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), args.lr)

    criterion = LossFn(**cfg['loss'], last_decoder_layer=model.decoder.out)
    criterion.to(device)

    # resume training
    if args.load_checkpoint:
        model, start_epoch, global_train_step = load_model_checkpoint(model, args.load_checkpoint, device)
        logger.global_train_step = global_train_step
        args.epochs += start_epoch
    else:
        start_epoch = 0

    # start run
    block_size = args.block_size
    logger.log_hparams({**cfg, **vars(args)})
    t_start = time.time()
    prev_loss = torch.inf
    for epoch in range(start_epoch, args.epochs):
        logger.init_epoch(epoch)
        print(f"Epoch [{epoch + 1} / {args.epochs}]")

        # train(model, data.train, optimizer, criterion, block_size, device)
        train(model, data.train, optimizer, criterion, device)


        # validate(model, data.val, criterion, block_size, device)
        validate(model, data.val, criterion, device)


        # logging
        output = ' - '.join([f'{k}: {v.avg:.4f}' for k, v in logger.epoch.items()])
        print(output)   
        loss = logger.epoch['val_loss'].avg
        if loss < prev_loss:
        # save logs and checkpoint
        # if (epoch + 1) % args.save_interval == 0 or (epoch + 1) == args.epochs:
            logger.save()
            if args.save_checkpoint:
                save_model_checkpoint(model, running_ckpt_dir, logger)
                save_model_checkpoint(criterion, running_ckpt_dir, logger, prefix='disc')
            prev_loss = loss

    elapsed_time = timer(t_start, time.time())
    print(f"Total training time: {elapsed_time}")


def train(model, train_loader, optimizer, criterion, device):
    model.train()

    for x, _ in tqdm(train_loader, desc="Training"):
        x = x.to(device)

        x_hat, z_e, z_q = model(x)
        # print(z_q.shape)
        # compute loss
        if criterion.disc_weight > 0 and logger.global_train_step > criterion.disc_warm_up_iters:
            # update generator
            loss, logs = criterion(x_hat, x, z_e, z_q, disc_training=True)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # update discriminator
            _, disc_logs = criterion.update_discriminator(x_hat, x)
            logs.update(disc_logs)
        else:
            loss, logs = criterion(x_hat, x, z_e, z_q)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        logger.log_metrics(logs, phase='train', aggregate=True, n=x.shape[0])

        if logger.global_train_step % 150 == 0:
            log2tensorboard_vqvae(logger, 'Train', logs.keys())
            ims = get_original_reconstruction_image(x, x_hat, n_ims=8)
            logger.tensorboard.add_image('Train: Original vs. Reconstruction', ims,
                                         global_step=logger.global_train_step, dataformats='HWC')

        logger.global_train_step += 1

    log2tensorboard_vqvae(logger, 'Train', logger.epoch.keys())

# def train(model, train_loader, optimizer, criterion, block_size, device):
#     model.train()
#     for x, _ in tqdm(train_loader, desc="Training"):
#         #num_blocks = x // block_size
#         x = x.to(device)
#         x_recon = torch.zeros_like(x).to(device)
#         optimizer.zero_grad()
#         for i in range(0, x.shape[-1], block_size):
#             for j in range(0, x.shape[-1], block_size):
#                 block = x[:, :, i:i+block_size, j:j+block_size]
#                 x_hat, z_e, z_q = model(block)
#                 x_recon[:, :, i:i+block_size, j:j+block_size] = x_hat
#                 # compute loss
#                 if criterion.disc_weight > 0 and logger.global_train_step > criterion.disc_warm_up_iters:
#                     # update generator
#                     loss, logs = criterion(x_hat, block, z_e, z_q, disc_training=True)
#                     optimizer.zero_grad()
#                     loss.backward()
#                     # update discriminator
#                     _, disc_logs = criterion.update_discriminator(x_hat, block)
#                     logs.update(disc_logs)
#                 else:
#                     loss, logs = criterion(x_hat, block, z_e, z_q)
#                     optimizer.zero_grad()
#                     loss.backward()
#         optimizer.step()
#         logger.log_metrics(logs, phase='train', aggregate=True, n=x.shape[0])

#         if logger.global_train_step % 150 == 0:
#             log2tensorboard_vqvae(logger, 'Train', logs.keys())
#             ims = get_original_reconstruction_image(x, x_recon, n_ims=8)
#             logger.tensorboard.add_image('Train: Original vs. Reconstruction', ims,
#                                         global_step=logger.global_train_step, dataformats='HWC')

#         logger.global_train_step += 1

#     log2tensorboard_vqvae(logger, 'Train', logger.epoch.keys())


@torch.no_grad()
def validate(model, val_loader, criterion, device):
    model.eval()

    is_first = True
    logs_keys = None
    for x, _ in tqdm(val_loader, desc="Validation"):
        x = x.to(device)

        x_hat, z_e, z_q = model(x)

        # compute loss
        loss, logs = criterion(x_hat, x, z_e, z_q)

        # logging
        logs = {'val_' + k: v for k, v in logs.items()}
        if logs_keys is None:
            logs_keys = logs.keys()
        logger.log_metrics(logs, phase='val', aggregate=True, n=x.shape[0])

        if is_first:
            is_first = False
            ims = get_original_reconstruction_image(x, x_hat, n_ims=8)
            logger.tensorboard.add_image('Val: Original vs. Reconstruction', ims,
                                         global_step=logger.global_train_step, dataformats='HWC')

    log2tensorboard_vqvae(logger, 'Val', logs_keys)
# def validate(model, val_loader, criterion, block_size, device):
#     model.eval()

#     is_first = True
#     logs_keys = None
#     for x, _ in tqdm(val_loader, desc="Validation"):
#         x = x.to(device)
#         x_recon = torch.zeros_like(x).to(device)
#         #prev_block = torch.rand_like(x[:, :, :block_size, :block_size]).to(device)
#         for i in range(0, x.shape[-1], block_size):
#             for j in range(0, x.shape[-1], block_size):
#                 block = x[:, :, i:i+block_size, j:j+block_size]
#                 x_hat, z_e, z_q = model(block)
#                 x_recon[:, :, i:i+block_size, j:j+block_size] = x_hat
#                 # compute loss
#                 loss, logs = criterion(x_hat, block, z_e, z_q)

#         # logging
#         logs = {'val_' + k: v for k, v in logs.items()}
#         if logs_keys is None:
#             logs_keys = logs.keys()
#         logger.log_metrics(logs, phase='val', aggregate=True, n=x.shape[0])

#         if is_first:
#             is_first = False
#             ims = get_original_reconstruction_image(x, x_recon, n_ims=8)
#             logger.tensorboard.add_image('Val: Original vs. Reconstruction', ims,
#                                             global_step=logger.global_train_step, dataformats='HWC')

#     log2tensorboard_vqvae(logger, 'Val', logs_keys)
#     return loss


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Exit training with keyboard interrupt!")
        logger.save()
        sys.exit(0)
