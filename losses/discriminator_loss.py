import torch
import torch.nn.functional as F


class DLogisticLoss(torch.nn.Module):
    def __init__(self):
        super(DLogisticLoss, self).__init__()

    def forward(self, real_pred, fake_pred):
        real_loss = F.softplus(-real_pred)
        fake_loss = F.softplus(fake_pred)
        return real_loss.mean() + fake_loss.mean()