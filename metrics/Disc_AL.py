# Average discriminator loss

try:
    from losses.discriminator_loss import DLogisticLoss
except:
    from paintbyword.losses.discriminator_loss import DLogisticLoss
import torch


class DiscAL(torch.nn.Module):
    def __init__(self, model):
        super(DiscAL, self).__init__()
        self.model = model

    def forward(self, images):
        losses = []
        for i in range(len(images)):
            loss_fn = DLogisticLoss()
            losses.append(loss_fn(self.model(images[i])))
        return torch.mean(torch.tensor(losses))
