import argparse

import torch
from torch.nn.functional import interpolate
import os


def generate_masks(amount, original):
    current_res = original.shape[1]
    current_mask = original.clone().unsqueeze(0).unsqueeze(0)
    result = {current_res: current_mask}
    for i in range(1, amount):
        next_mask = interpolate(current_mask, current_res, mode='nearest')
        result[current_res] = next_mask
        current_mask = next_mask
        current_res //= 2
    return result


def get_args(dir_name):
    parser = argparse.ArgumentParser(description='Model config')
    parser.add_argument('--config', type=str, dest='config_path',
                        help='path to config', default=os.path.join(dir_name, 'configs/lsun_churches.json'))

    parser.add_argument('--weights', type=str, dest='stylegan_weights',
                        help='path to StyleGANv2 pytorch weights', required=True)

    parser.add_argument('--results', type=str, dest='results_path', help='Path to save results', required=True)
    parser.add_argument('--mask', type=str, dest='mask_path', help='Path to .npy file with the mask array',
                        required=True)
    parser.add_argument('--latent_path', type=str, dest='latent_path',
                        help='Path to .pt file with the input image latent code', required=True)
    parser.add_argument('--text', type=str, dest='text_input')
    parser.add_argument('--seed', type=int, default=None, dest='seed')

    args = parser.parse_args()
    return args


def extract_latent(latent, shape):
    if not isinstance(latent, torch.Tensor):
        latent = list(latent.items())[0][1]['latent']
    actual_shape = latent.shape
    if len(actual_shape) < 3 or actual_shape[1] != shape:
        return latent.repeat((1, shape, 1))
    return latent
