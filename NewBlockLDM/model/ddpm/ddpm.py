from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from model import VQGANLight
from model.ddpm.beta_schedule import BetaSchedule


class DDPM(nn.Module):
    def __init__(
            self,
            eps_model: nn.Module,
            vae_model: nn.Module,
            beta_1: float,
            beta_2: float,
            beta_schedule: str,
            n_steps: int,
            loss_function: str,
            loss_against: str,
    ):
        """
        Implementation of a simple Denoising Diffusion Probabilistic Model (DDPM) like presented in
        https://arxiv.org/abs/2006.11239

        Args:
            eps_model: unet as neural backbone of the DDPM
            vae_model: variational autoencoder model for encoding and decoding
            beta_1: lower variance bound for the beta schedule
            beta_2: upper variance bound for the beta schedule
            beta_schedule: beta schedule that defines how noise should be added at each timestep
            n_steps: number of timesteps
            loss_function: loss function for training
        """
        super(DDPM, self).__init__()
        self.n_steps = n_steps
        self.eps_model = eps_model
        self.vae_model = vae_model

        # we don't want to train any parameters of the vae model
        for param in vae_model.parameters():
            param.requires_grad = False

        if not beta_1 < beta_2 < 1.0:
            raise ValueError(f"beta1: {beta_1} < beta2: {beta_2} < 1.0 not fulfilled")

        available_beta_schedules = ["linear", "quadratic", "sigmoid", "cosine"]
        if beta_schedule not in available_beta_schedules:
            raise ValueError(f"Beta schedule should be one of the following: {available_beta_schedules}")

        available_loss_functions = ["l1", "l2", "huber"]
        if loss_function not in available_loss_functions:
            raise ValueError(f"Loss function should be one of the following: {available_loss_functions}")
        self.loss_function = loss_function

        # define beta schedule
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.betas = BetaSchedule(self.beta_1, self.beta_2, beta_schedule, self.n_steps).values

        # define alphas
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, axis=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        self.count = 0
        self.loss_against = loss_against

    def extract(self, a, t, x_shape):
        """
        extracts an appropriate t index for a batch of indices
        """
        batch_size = t.shape[0]
        out = a.gather(-1, t.cpu())
        return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)

    @torch.no_grad()
    def encode(self, x: torch.Tensor):
        """
        Encodes and quantizes an input image

        Args:
            x: the image to encode
        Returns:
            x: encoded and quantized image
        """
        x = self.vae_model.encode(x)
        x = self.vae_model.quantize(x)

        return x

    @torch.no_grad()
    def decode(self, x: torch.Tensor):
        """
        Decode latent representation to an image

        Args:
            x: latent representation to decode
        Returns:
            x_hat: decoded latent representation
        """
        x_hat = self.vae_model.decode(x)

        return x_hat

    def q_sample(self, x_start, t, noise=None):
        """
        forward diffusion process

        Args:
            x_start: start value for the forward process
            t: current timestep
            noise: noise to add to the start value
        Returns:
            x_start with added noise corresponding to the current timestep
        """
        if noise is None:
            noise = torch.randn_like(x_start)

        sqrt_alphas_cumprod_t = self.extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = self.extract(
            self.sqrt_one_minus_alphas_cumprod, t, x_start.shape
        )

        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

    def p_losses(self, x_start, x_cond, position, low_res_cond = None, noise=None):
        """
        runs a forward step and calculates the loss

        Args:
            x_start: start value for the forward process
            noise: noise to add to the start value
        Returns:
            loss between the noise and the predicted noise of the epsilon model
        """
        # print("x_start_shape:" , x_start.shape)
        # print("x_cond_shape:" , x_cond.shape)
        # print("position_shape:" , position.shape)
        # print("low_res_cond_shape:" , low_res_cond.shape)
        x_start = self.encode(x_start)
        x_cond = self.encode(x_cond)
        if low_res_cond is not None:
            low_res_cond = self.encode(low_res_cond)
        # print("after encoding x_start_shape:" , x_start.shape)
        # print("after encoding x_cond_shape:" , x_cond.shape)
        # print("after encoding low_res_cond_shape:" , low_res_cond.shape)
        if noise is None:
            noise = torch.randn_like(x_start)

        # t = torch.randint(0, self.n_steps, (x_start.shape[0],), dtype=torch.int64).to(x_start.device)  # t ~ Uniform({1, ..., T})
        random_time = torch.randint(0, self.n_steps, (1,)).item()
        t = torch.full((x_start.shape[0],), random_time, dtype=torch.int64).to(x_start.device)

        x_noisy = self.q_sample(x_start, t, noise)

        predicted_noise = self.eps_model(x_noisy, x_cond, t, position, low_res_cond)
        # if random_time <= 5:
        #     x_recon = self.reconstruction_loop(x_start, x_noisy, x_cond, position, t)
        #     self.count += 1
        #     return self.calculate_loss(noise, predicted_noise, x_start, x_recon)
        return self.calculate_loss(noise, predicted_noise)
    def p_losses2(self, x_start, x_cond, position, noise=None):
            """
            runs a forward step and calculates the loss

            Args:
                x_start: start value for the forward process
                noise: noise to add to the start value
            Returns:
                loss between the noise and the predicted noise of the epsilon model
            """
            if noise is None:
                noise = torch.randn_like(x_start)

            random_time = torch.randint(0, self.n_steps, (1,)).item()
            t = torch.full((x_start.shape[0],), random_time, dtype=torch.int64).to(x_start.device)

            x_noisy = self.q_sample(x_start, t, noise)
            x_noisy = torch.cat([x_cond, x_noisy], dim = 1)
            predicted_noise = self.eps_model(x_noisy, t, position)
            if self.loss_against == 'x0':
                return self.calculate_loss(x_start, predicted_noise)
            return self.calculate_loss(noise, predicted_noise)

    def calculate_loss(self, noise, predicted_noise, x_start = None, x_recon = None):
        """
        calculates the loss according to the defined loss function

        Args:
            noise: ground truth
            predicted_noise: prediction
        Returns:
            the calculated loss between ground truth and prediction
        """
        if self.loss_function == "l1":
            loss = F.l1_loss(noise, predicted_noise)
        elif self.loss_function == "l2":
            loss = F.mse_loss(noise, predicted_noise) #, reduction = 'sum')/noise.size(0)
        elif self.loss_function == "huber":
            loss = F.smooth_l1_loss(noise, predicted_noise)
        else:
            raise NotImplementedError()
        if x_start is not None:
            return loss + F.mse_loss(x_start, x_recon)
        return loss

    def reconstruction_loop(self, x, x_noisy, x_cond, position, t):
        device = next(self.eps_model.parameters()).device

        b = x.shape[0]
        # create noise
        img = x_noisy
        # print(img.shape, cond_block.shape)
        for t_index in reversed(range(0, t[0])):
            img = self.p_sample(img, x_cond, position, torch.full((b,), t_index, device=device, dtype=torch.int64), t_index)
        return img
    
    @torch.no_grad()
    def p_sample(self, x, x_prev, position, t, t_index):
        """
        samples an image from the latent space

        Args:
            x: latent tensor to draw the image for
            t_index: current timestep
        Returns:
            the sampled image
        """
        betas_t = self.extract(self.betas, t, x.shape)
        sqrt_one_minus_alphas_cumprod_t = self.extract(
            self.sqrt_one_minus_alphas_cumprod, t, x.shape
        )
        sqrt_recip_alphas_t = self.extract(self.sqrt_recip_alphas, t, x.shape)

        # Equation 11 in https://arxiv.org/abs/2006.11239
        # Use our model (noise predictor) to predict the mean

        model_mean = sqrt_recip_alphas_t * (
                x - betas_t * self.eps_model(torch.cat([x_prev, x], dim=1), t, position) / sqrt_one_minus_alphas_cumprod_t
        )
        if t_index == 0:
            return model_mean
        else:
            posterior_variance_t = self.extract(self.posterior_variance, t, x.shape)
            noise = torch.randn_like(x)
            # Algorithm 2 line 4:
            return model_mean + torch.sqrt(posterior_variance_t) * noise
        
    @torch.no_grad()
    def p_sample_loop(self, cond_block, position, shape, low_res_cond = None, sample_step=None):
        """
        Implements Algorithm 2 of https://arxiv.org/abs/2006.11239 for sampling

        Args:
            shape: target shape of the sampled images
            sample_step: if not None sampling only returns samples from specified timestep
        Returns:
            sampled images
        """
        device = next(self.eps_model.parameters()).device

        b = shape[0]
        # create noise
        img = torch.randn(shape, device=device)
        imgs = []
        # print(img.shape, cond_block.shape)
        for i in tqdm(reversed(range(0, self.n_steps)), desc='sampling loop time step', total=self.n_steps):
            img = self.p_sample(img, cond_block, position, torch.full((b,), i, device=device, dtype=torch.int64), i)
            if sample_step is not None and i == sample_step:
                imgs.append(img)
            elif sample_step is None:
                imgs.append(img)
        return imgs

    @torch.no_grad()
    def sample(self, image_size, cond_block, position, batch_size=16, channels=3, sample_step=None):
        """
        sampling from the latent space

        Args:
            image_size: target size of the sampled images
            batch_size: number of sampled images
            channels: number of channels for the sampled images
            sample_step: if not None sampling only returns samples from specified timestep
        Returns:
            sampled images
        """
        return  self.p_sample_loop(cond_block, position, shape=(batch_size, channels, image_size, image_size), sample_step=sample_step)

