from torch.nn.functional import interpolate


def generate_masks(amount, original):
    current_res = original.shape[1] // 2
    current_mask = original.clone()
    result = {current_res: current_mask}
    for i in range(1, amount):
        next_mask = interpolate(current_mask, current_res, mode='nearest')
        result[current_res] = next_mask
        current_mask = next_mask
        current_res //= 2
