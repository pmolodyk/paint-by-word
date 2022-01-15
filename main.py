import argparse
import math
import os
from pathlib import Path

import json
import numpy as np
import torch
import torchvision
from tqdm import tqdm

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

args = parser.parse_args()
config_path = args.config_path
stylegan_weights_path = args.stylegan_weights
results_path = args.results_path
Path(results_path).mkdir(parents=True, exist_ok=True)

# StyleGANv2
config = json.load(open(config_path, 'r'))
stylegan_size = config['stylegan']['size']
stylegan_style_dim = config['stylegan']['style_dim']
stylegan_n_mlp = config['stylegan']['n_mlp']
latent_chunk_num = config['stylegan']['latent_chunk_num']   # TODO check if this is necessary

mask = torch.Tensor(np.load(args.mask_path)).cuda()
upscale_layers_num = int(math.log(stylegan_size, 2)) - 2
mask_by_resolution = generate_masks(upscale_layers_num, mask)

w1 = torch.zeros(1, latent_chunk_num, stylegan_style_dim)
stylegan_generator = Generator(stylegan_size, stylegan_style_dim, stylegan_n_mlp, mask_by_resolution).cuda()
stylegan_generator.load_state_dict(torch.load(stylegan_weights_path)["g_ema"], strict=False)
stylegan_generator.eval()

# Optimization settings
opt_steps = config['optimization']['opt_steps']
lr = config['optimization']['lr']

if 'generate' in config and not config['generate']:
    raise RuntimeError('Edit not supported yet, set generate to true in config')  # TODO
else:
    mean_latent = stylegan_generator.mean_latent(4096)
    latent = mean_latent.detach().clone().repeat(1, latent_chunk_num, 1).detach().clone()
    current_image, _ = stylegan_generator([latent], input_is_latent=True, randomize_noise=False)
    torchvision.utils.save_image(current_image, os.path.join(results_path, 'initial_image.jpg'), normalize=True, range=(-1, 1))

w_add = torch.zeros(latent.shape).cuda()
w_add.requires_grad = True
