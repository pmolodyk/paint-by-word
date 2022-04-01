import json
import math
import os
from pathlib import Path

import clip
import lpips
import numpy as np
import torch
from torch.optim import Adam
import torchvision
from tqdm import tqdm

from losses.CLIP_loss import CLIPLoss
from losses.img_loss import ImgLoss
from losses.discriminator_loss import DLogisticLoss
from models.styleganv2.model import Generator, Discriminator
from utils import generate_masks, get_args

dir_name = os.path.dirname(__file__)

# Args parser
args = get_args(dir_name)
config_path = args.config_path
latent_path = args.latent_path
stylegan_weights_path = args.stylegan_weights
results_path = args.results_path
Path(results_path).mkdir(parents=True, exist_ok=True)

# Random
seed = args.seed
if seed is not None:
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    np.random.seed(seed)

# StyleGANv2
config = json.load(open(config_path, 'r'))
stylegan_size = config['stylegan']['size']
stylegan_style_dim = config['stylegan']['style_dim']
stylegan_n_mlp = config['stylegan']['n_mlp']
no_split_layers_num = config['optimization']['no_split_layers_num']

mask = torch.Tensor(np.load(args.mask_path)).cuda()
latent = torch.load(latent_path).cuda()
upscale_layers_num = int(math.log(stylegan_size, 2)) - 1
mask_by_resolution = generate_masks(upscale_layers_num, mask)

w1 = latent.clone().cuda()

ckpt = torch.load(stylegan_weights_path)
stylegan_generator = Generator(stylegan_size, stylegan_style_dim, stylegan_n_mlp, mask_by_resolution,
                               no_split_layers_num=no_split_layers_num).cuda()
stylegan_generator.load_state_dict(ckpt["g_ema"], strict=False)
stylegan_generator.eval()

stylegan_discriminator = Discriminator(stylegan_size)
stylegan_discriminator.load_state_dict(ckpt["d"], strict=False)
stylegan_discriminator.eval()

# Optimization settings
opt_steps = config['optimization']['opt_steps']
lr = config['optimization']['lr']
sem_lam = config['optimization']['semantic_loss_lambda']
l2_lam = config['optimization']['l2_loss_lambda']
img_lam = config['optimization']['img_loss_lambda']
latent_prox_lam = config['optimization']['latent_prox_lam']
global_lpips_lam = config['optimization']['global_lpips_lam']
discriminator_loss_lam = config['optimization']['discriminator_loss_lam']
clip_loss = CLIPLoss(stylegan_size)
image_loss = ImgLoss(lam=l2_lam)
latent_proximity_loss = torch.nn.MSELoss()
global_lpips_loss = lpips.LPIPS(net='vgg').cuda()
discriminator_loss = DLogisticLoss()

text_tokenized = torch.cat([clip.tokenize(args.text_input)]).cuda()
with torch.no_grad():
    current_image, _ = stylegan_generator([latent], w1, input_is_latent=True, randomize_noise=False)
torchvision.utils.save_image(current_image, os.path.join(results_path, 'initial_image.jpg'), normalize=True, range=(-1, 1))
external_region = current_image.clone() * (1 - mask)
initial_image = current_image.clone().detach()

for param in stylegan_generator.parameters():
    param.requires_grad = False

for param in stylegan_discriminator.parameters():
    param.requires_grad = False

# Optimization
w1.requires_grad = True
optimizer = Adam([w1], lr=lr)
for step in tqdm(range(opt_steps)):
    current_image, _ = stylegan_generator([latent], w1, input_is_latent=True, randomize_noise=False)
    critic_verdict_fake = stylegan_discriminator(current_image)
    critic_verdict_real = stylegan_discriminator(initial_image)

    l_sem = clip_loss(current_image, text_tokenized)
    l_img = image_loss(external_region, current_image * (1 - mask))
    l_prox = latent_proximity_loss(latent, w1)
    l_global = global_lpips_loss(initial_image, current_image)
    l_disc = discriminator_loss(critic_verdict_real, critic_verdict_fake)
    loss = l_sem * sem_lam \
            + l_img * img_lam \
            + l_prox * latent_prox_lam \
            + l_global * global_lpips_lam \
            + l_disc * discriminator_loss_lam

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

torchvision.utils.save_image(current_image, os.path.join(results_path, 'final_image.jpg'), normalize=True, range=(-1, 1))
