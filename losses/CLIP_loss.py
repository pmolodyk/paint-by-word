# Borrowed from https://github.com/orpatashnik/StyleCLIP/blob/main/criteria/clip_loss.py

import torch
import clip


class CLIPLoss(torch.nn.Module):
    def __init__(self, resolution, mode):
        super(CLIPLoss, self).__init__()
        self.mode = mode
        self.model, self.preprocess = clip.load("ViT-B/32", device="cuda")
        if mode == 'product':
            self.upsample = torch.nn.Upsample(scale_factor=7)
            self.avg_pool = torch.nn.AvgPool2d(kernel_size=resolution // 32)
        elif mode == 'crop':
            self.upsample = torch.nn.Upsample(size=224)
        else:
            raise AttributeError('Invalid CLIP loss mode')

    def forward(self, image, text):
        image = self.upsample(image)
        if self.mode == "product":
            image = self.avg_pool(image)
        similarity = 1 - self.model(image, text)[0] / 100
        return similarity
