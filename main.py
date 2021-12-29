import argparse

import json
import torch

from models.styleganv2.model import Generator

#Args parser
parser = argparse.ArgumentParser(description='Model config')
parser.add_argument('--config', type=str, dest='config_path',
                    help='path to config', default='configs/lsun_churches.json')

parser.add_argument('--weights', type=str, dest='stylegan_weights',
                    help='path to StyleGANv2 pytorch weights', required=True)

args = parser.parse_args()
config_path = args.config_path
stylegan_weights_path = args.stylegan_weights

# StyleGANv2
config = json.load(open(config_path, 'r'))
stylegan_size = config['stylegan']['size']
stylegan_style_dim = config['stylegan']['style_dim']
stylegan_n_mlp = config['stylegan']['n_mlp']

stylegan_generator = Generator(stylegan_size, stylegan_style_dim, stylegan_n_mlp)
stylegan_generator.load_state_dict(torch.load(stylegan_weights_path)["g_ema"], strict=False)
