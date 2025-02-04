# RISSOLE: Parameter-efficient Diffusion Models via Block-wise Generation and Retrieval-Guidance

## Tensorboard

In order to monitor the losses and visualizations just `cd` into the repo and run
```
tensorboard --logdir=logs
```

## VQ-GAN

Implementation of VQ-GAN ([paper](https://arxiv.org/abs/2012.09841)).

#### Usage


```
python3 train_vqgan.py --name myExp --epochs 2 --config configs/vqgan.yaml
```

To first debug the code with `CIFAR10` just run

```
python3 train_vqgan.py --epochs 2 --config configs/vqgan.yaml --debug
```

#### Model Settings

You can specify the model settings in the `vqgan.yaml` config file. The length of the channels list in the config files also determines the down-scaling of the input image. For example, a list with two channels (eg [32, 64]) down-samples the image by a factor of 4.

```
model:
  autoencoder_cfg:
    in_channels: 3
    channels:
      - 32
      - 64
    dim_keys: 64
    n_heads: 4
  latent_dim: 32
  n_embeddings: 512
```

#### Losses

You can specify which losses to use and which weights for which loss in the `vqgan.yaml` config file.

```
loss:
  rec_loss_type: 'L1'
  perceptual_weight: 0.1
  codebook_weight: 0.9
  commitment_weight: 0.25
  disc_weight: 0.1
  disc_in_channels: 3
  disc_n_layers: 4
  disc_warm_up_iters: 5000
  disc_res_blocks: False
```
