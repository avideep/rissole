import os
import pathlib
import sys
import time
import yaml
import argparse
from tqdm import tqdm
from datetime import datetime
import torch
import torch.nn as nn
from model import VAE, IntroVAE
from torch.autograd import Variable
from model.losses import LossFn
from utils.logger import Logger
import torch.nn.functional as F
from utils.helpers import timer
from utils.helpers import load_model_checkpoint, save_model_checkpoint
from utils.helpers import log2tensorboard_vqvae
from utils.helpers import count_parameters
from utils.visualization import get_original_reconstruction_image
from dataloader import PlantNet, CelebA, CelebAHQ

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

CHECKPOINT_DIR = os.path.join(pathlib.Path(__file__).parent.resolve(), 'checkpoints')
LOG_DIR = os.path.join(pathlib.Path(__file__).parent.resolve(), 'logs')
TIMESTAMP = datetime.now().strftime('%y-%m-%d_%H%M%S')

parser = argparse.ArgumentParser(description="PyTorch Auxilliary VAE Training")
parser.add_argument('--name', '-n', default='introVAE',
                    type=str, metavar='NAME', help='Model name and folder where logs are stored')
parser.add_argument('--epochs', default=500,
                    type=int, metavar='N', help='Number of epochs to run (default: 2)')
parser.add_argument('--batch-size', default=16, metavar='N',
                    type=int, help='Mini-batch size (default: 64)')
parser.add_argument('--image-size', default=256, metavar='N',
                    type=int, help='Size that images should be resized to before processing (default: 128)')
parser.add_argument('--beta', default=1, metavar='N',
                    type=int, help='beta in beta-VAE')
parser.add_argument('--num-workers', default=0, metavar='N',
                    type=int, help='Number of workers for the dataloader (default: 0)')
parser.add_argument('--config', default='configs/vae.yaml',
                    metavar='PATH', help='Path to model config file (default: configs/vae.yaml)')
parser.add_argument('--data-config', default='configs/data_se.yaml',
                    metavar='PATH', help='Path to model config file (default: configs/data_se.yaml)')
parser.add_argument('--debug', action='store_true',
                    help='If true, trains on CIFAR10')
parser.add_argument('--gpus', default=0, type=int,
                    nargs='+', metavar='GPUS', help='If GPU(s) available, which GPU(s) to use for training.')
parser.add_argument('--ckpt-save', default=True, action='store_true',
                    dest='save_checkpoint', help='Save checkpoints to folder')
parser.add_argument('--load-ckpt', default=None, metavar='PATH',
                    dest='load_checkpoint', help='Load model checkpoint and continue training')
parser.add_argument('--log-save-interval', default=5, type=int, metavar='N',
                    dest='save_interval', help="Interval in which logs are saved to disk (default: 5)")
parser.add_argument('--lr_e', type=float, default=0.0002, help='learning rate of the encoder, default=0.0002')
parser.add_argument('--lr_g', type=float, default=0.0002, help='learning rate of the generator, default=0.0002')
parser.add_argument("--num_vae", type=int, default=0, help="the epochs of pretraining a VAE, Default=0")
parser.add_argument("--weight_neg", type=float, default=1.0, help="Default=1.0")
parser.add_argument("--weight_rec", type=float, default=1.0, help="Default=1.0")
parser.add_argument("--weight_kl", type=float, default=1.0, help="Default=1.0")
parser.add_argument("--m_plus", type=float, default=100.0, help="the margin in the adversarial part, Default=100.0")
parser.add_argument('--channels', default="64, 128, 256, 512, 512, 512", type=str, help='the list of channel numbers')
parser.add_argument("--hdim", type=int, default=512, help="dim of the latent code, Default=512")
parser.add_argument("--save_iter", type=int, default=1, help="Default=1")
parser.add_argument("--test_iter", type=int, default=1000, help="Default=1000")
parser.add_argument('--nrow', type=int, help='the number of images in each row', default=8)
parser.add_argument('--lr', type=float, default=0.0002, help='learning rate, default=0.0002')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument("--momentum", default=0.9, type=float, help="Momentum, Default: 0.9")
parser.add_argument('--clip', type=float, default=100, help='the threshod for clipping gradient')
parser.add_argument("--step", type=int, default=500, help="Sets the learning rate to the initial LR decayed by momentum every n epochs, Default: n=500")
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--outf', default='results/', help='folder to output images and model checkpoints')
parser.add_argument('--manualSeed', type=int, help='manual seed')
parser.add_argument('--tensorboard', action='store_true', help='enables tensorboard')
parser.add_argument("--pretrained", default="", type=str, help="path to pretrained model (default: none)")

logger = Logger(LOG_DIR)
torch.autograd.set_detect_anomaly(True)
def loss_fn(x, recon_x, mu, logvar, beta):
    log = {}
    recon_loss = nn.MSELoss(reduction='sum')(recon_x, x)
    log['recons_loss'] = recon_loss.item() / x.shape[0]
    kl_diverge = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    log['kl_loss'] = kl_diverge.item() / x.shape[0]
    loss = (recon_loss + beta * kl_diverge) / x.shape[0]
    log['loss'] = loss.item()
    return loss, log  

def train(model, train_loader, optimizerE, optimizerG, beta, device, args):
    model.train()

    logs_keys = None
    for x, _ in tqdm(train_loader, desc="Training"):
        log = {}
        x = x.to(device)
        if len(x.size()) == 3:
                x = x.unsqueeze(0)
                
        batch_size = x.size(0)
        
        noise = Variable(torch.zeros(batch_size, args.hdim).normal_(0, 1)).cuda() 
            
        real= Variable(x).cuda() 
        
        # info = "\n====> Cur_iter: [{}]: Epoch[{}]({}/{}): time: {:4.4f}: ".format(cur_iter, epoch, iteration, len(train_data_loader), time.time()-start_time)
        
        # loss_info = '[loss_rec, loss_margin, lossE_real_kl, lossE_rec_kl, lossE_fake_kl, lossG_rec_kl, lossG_fake_kl,]'
            
        #=========== Update E ================ 
        fake = model.sample(noise)            
        real_mu, real_logvar, z, rec = model(real)
        rec_mu, rec_logvar = model.encode(rec.detach())
        fake_mu, fake_logvar = model.encode(fake.detach())
        
        loss_rec =  model.reconstruction_loss(rec, real, True)
        log['loss_rec'] = loss_rec.item()
        lossE_real_kl = model.kl_loss(real_mu, real_logvar).mean()
        lossE_rec_kl = model.kl_loss(rec_mu, rec_logvar).mean()
        lossE_fake_kl = model.kl_loss(fake_mu, fake_logvar).mean()            
        loss_margin = lossE_real_kl + \
                    (F.relu(args.m_plus-lossE_rec_kl) + \
                    F.relu(args.m_plus-lossE_fake_kl)) * 0.5 * args.weight_neg
        
                    
        lossE = loss_rec  * args.weight_rec + loss_margin * args.weight_kl
        log['lossE'] = lossE.item()
        optimizerG.zero_grad()
        optimizerE.zero_grad()       
        lossE.backward(retain_graph=True)
        # nn.utils.clip_grad_norm(model.encoder.parameters(), 1.0)            
        optimizerE.step()
        
        #========= Update G ==================           
        rec_mu, rec_logvar = model.encode(rec)
        fake_mu, fake_logvar = model.encode(fake)
        
        lossG_rec_kl = model.kl_loss(rec_mu, rec_logvar).mean()
        lossG_fake_kl = model.kl_loss(fake_mu, fake_logvar).mean()
        
        lossG = (lossG_rec_kl + lossG_fake_kl)* 0.5 * args.weight_kl      
        log['lossG'] = lossG.item()
                    
        # optimizerG.zero_grad()
        lossG.backward()
        # nn.utils.clip_grad_norm(model.decoder.parameters(), 1.0)
        optimizerG.step()
        if logs_keys is None:
            logs_keys = log.keys()

        logger.log_metrics(log, phase='train', aggregate=True, n=x.shape[0])

        if logger.global_train_step % 150 == 0:
            log2tensorboard_vqvae(logger, 'Train', logs_keys)
            ims = get_original_reconstruction_image(x, rec, n_ims=8)
            logger.tensorboard.add_image('Train: Original vs. Reconstruction', ims,
                                         global_step=logger.global_train_step, dataformats='HWC')

        logger.global_train_step += 1

    log2tensorboard_vqvae(logger, 'Train', logs_keys)


@torch.no_grad()
def validate(model, val_loader, beta, device):
    model.eval()

    is_first = True
    logs_keys = None
    for x, _ in tqdm(val_loader, desc="Validation"):
        x = x.to(device)

        mu, logvar, _, recon_x = model(x)

        # compute loss
        _, logs = loss_fn(x, recon_x, mu, logvar, beta)

        # logging
        logs = {'val_' + k: v for k, v in logs.items()}
        if logs_keys is None:
            logs_keys = logs.keys()
        logger.log_metrics(logs, phase='val', aggregate=True, n=x.shape[0])

        if is_first:
            is_first = False
            ims = get_original_reconstruction_image(x, recon_x, n_ims=8)
            logger.tensorboard.add_image('Val: Original vs. Reconstruction', ims,
                                         global_step=logger.global_train_step, dataformats='HWC')

    log2tensorboard_vqvae(logger, 'Val', logs_keys)

def main():
    args = parser.parse_args()
    for name, val in vars(args).items():
        print("{:<16}: {}".format(name, val))

    # setup paths and logging
    args.name = 'vae/' + args.name
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
    if args.debug:
        # data = CIFAR10(args.batch_size)
        data = CelebA(args.batch_size, args.image_size)
    else:
        data = CelebAHQ(args.batch_size)


    # read config file for model
    cfg = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)

    # create model and optimizer
    # model = VAE(**cfg['model'])
    model = IntroVAE(**cfg['model']).cuda()    

    print("{:<16}: {}".format('model params', count_parameters(model)))
    model.to(device)

    # optimizer = torch.optim.Adam(model.parameters(), args.lr)
    optimizerE = torch.optim.Adam(model.encoder.parameters(), lr=args.lr_e)
    optimizerG = torch.optim.Adam(model.decoder.parameters(), lr=args.lr_g)

    # resume training
    if args.load_checkpoint:
        model, start_epoch, global_train_step = load_model_checkpoint(model, args.load_checkpoint, device)
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
        print(f"Epoch [{epoch + 1} / {args.epochs}]")

        train(model, data.train, optimizerE, optimizerG, args.beta, device, args)

        validate(model, data.val, args.weight_kl, device)

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
                # save_model_checkpoint(criterion, running_ckpt_dir, logger, prefix='disc')
            prev_loss = loss

    elapsed_time = timer(t_start, time.time())
    print(f"Total training time: {elapsed_time}")
if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Exit training with keyboard interrupt!")
        logger.save()
        sys.exit(0)
