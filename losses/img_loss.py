import lpips
import torch

class ImgLoss(torch.nn.Module):
    def __init__(self, lam=1, lpips_mode='vgg'):   # Mode change not recommended
        super(ImgLoss, self).__init__()
        self.lpips = lpips.LPIPS(net=lpips_mode)
        self.lam = lam
        self.l2 = torch.nn.MSELoss()

    def forward(self, im0, im1):
        return self.l2(im0, im1) + self.lam * self.lpips(im0, im1)