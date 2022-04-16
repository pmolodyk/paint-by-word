# CLIP Average Loss Improvement metric

from losses.CLIP_loss import CLIPLoss
import torch


class CLIPALI(torch.nn.Module):
    def __init__(self):
        super(CLIPALI, self).__init__()

    def forward(self, images, originals, text):
        if len(images) != len(originals):
            raise ValueError('Supply an equal number of edits and originals')
        loss_diffs = []
        for i in range(len(images)):
            resolution = images[i].shape[-1]
            clip_loss = CLIPLoss(resolution, "crop")
            loss_diffs.append(clip_loss(originals[i], text) - clip_loss(images[i], text))
        return torch.mean(torch.tensor(loss_diffs))
