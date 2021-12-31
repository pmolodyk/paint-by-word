import argparse
import os
from pathlib import Path

import json
import torch
from tqdm import tqdm
from PIL import Image

from models.styleganv2.model import Generator

dir_name = os.path.dirname(__file__)

# Args parser
parser = argparse.ArgumentParser(description='Model config')
parser.add_argument('--config', type=str, dest='config_path',
                    help='path to config', default=os.path.join(dir_name, 'configs/lsun_churches.json'))

parser.add_argument('--weights', type=str, dest='stylegan_weights',
                    help='path to StyleGANv2 pytorch weights', required=True)

parser.add_argument('--results', type=str, dest='results_path', help='Path to save results', required=True)

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

stylegan_generator = Generator(stylegan_size, stylegan_style_dim, stylegan_n_mlp).cuda()
stylegan_generator.load_state_dict(torch.load(stylegan_weights_path)["g_ema"], strict=False)
stylegan_generator.eval()

# Optimization settings
opt_steps = config['optimization']['opt_steps']
lr = config['optimization']['lr']

if 'generate' in config and not config['generate']:
    raise RuntimeError('Edit not supported yet, set generate to true in config')  # TODO
else:
    mean_latent = stylegan_generator.mean_latent(4096)
    latent = mean_latent.detach().clone().repeat(1, 18, 1).detach().clone()
    current_image, _ = stylegan_generator([latent], input_is_latent=True, randomize_noise=False)
    Image.fromarray(current_image.detach().clone().cpu().numpy()).save(os.path.join(results_path, 'initial_image.jpg'))
    latent.requires_grad = True
