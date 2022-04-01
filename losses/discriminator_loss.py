import torch
import torch.nn.functional as F


class DLogisticLoss(torch.nn.Module):
    def __init__(self):
        super(DLogisticLoss, self).__init__()

    def forward(self, fake_pred):
        fake_loss = F.softplus(fake_pred)
        return -fake_loss.mean()