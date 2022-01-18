import argparse
import json
import math
import os
from pathlib import Path

import clip
import numpy as np
import torch
from torch.optim import Adam
import torchvision
from tqdm import tqdm

from losses.CLIP_loss import CLIPLoss
from losses.img_loss import ImgLoss
from models.styleganv2.model import Generator
from utils import generate_masks

dir_name = os.path.dirname(__file__)

# Args parser
parser = argparse.ArgumentParser(description='Model config')
parser.add_argument('--config', type=str, dest='config_path',
                    help='path to config', default=os.path.join(dir_name, 'configs/lsun_churches.json'))

parser.add_argument('--weights', type=str, dest='stylegan_weights',
                    help='path to StyleGANv2 pytorch weights', required=True)

parser.add_argument('--results', type=str, dest='results_path', help='Path to save results', required=True)
parser.add_argument('--mask', type=str, dest='mask_path', help='Path to .npy file with the mask array', required=True)
parser.add_argument('--latent_path', type=str, dest='latent_path',
                    help='Path to .pt file with the input image latent code', required=True)
parser.add_argument('--text', type=str, dest='text_input')

args = parser.parse_args()
config_path = args.config_path
latent_path = args.latent_path
stylegan_weights_path = args.stylegan_weights
results_path = args.results_path
Path(results_path).mkdir(parents=True, exist_ok=True)

# StyleGANv2
config = json.load(open(config_path, 'r'))
stylegan_size = config['stylegan']['size']
stylegan_style_dim = config['stylegan']['style_dim']
stylegan_n_mlp = config['stylegan']['n_mlp']

mask = torch.Tensor(np.load(args.mask_path)).cuda()
latent = torch.load(latent_path).cuda()
upscale_layers_num = int(math.log(stylegan_size, 2)) - 1
mask_by_resolution = generate_masks(upscale_layers_num, mask)

w1 = latent.clone().cuda()
stylegan_generator = Generator(stylegan_size, stylegan_style_dim, stylegan_n_mlp, mask_by_resolution).cuda()
stylegan_generator.load_state_dict(torch.load(stylegan_weights_path)["g_ema"], strict=False)
stylegan_generator.eval()

# Optimization settings
opt_steps = config['optimization']['opt_steps']
lr = config['optimization']['lr']
l2_lam = config['optimization']['l2_loss_lambda']
img_lam = config['optimization']['img_loss_lambda']
clip_loss = CLIPLoss(stylegan_size)
image_loss = ImgLoss(lam=l2_lam)

text_tokenized = torch.cat([clip.tokenize(args.text_input)]).cuda()
with torch.no_grad():
    current_image, _ = stylegan_generator([latent], w1, input_is_latent=True, randomize_noise=False)
torchvision.utils.save_image(current_image, os.path.join(results_path, 'initial_image.jpg'), normalize=True, range=(-1, 1))
external_region = current_image.clone() * (1 - mask)

# Optimization
w1.requires_grad = True
optimizer = Adam([w1], lr=lr)
for step in tqdm(range(opt_steps)):
    current_image, _ = stylegan_generator([latent], w1, input_is_latent=True, randomize_noise=False)
    l_sem = clip_loss(current_image, text_tokenized)
    l_img = image_loss(external_region, current_image * (1 - mask))
    loss = l_sem + img_lam * l_img

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

torchvision.utils.save_image(current_image, os.path.join(results_path, 'final_image.jpg'), normalize=True, range=(-1, 1))
