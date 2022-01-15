from torch.nn.functional import interpolate


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
