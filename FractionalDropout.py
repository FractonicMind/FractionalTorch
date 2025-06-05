import torch
import torch.nn as nn

class FractionalDropout(nn.Module):
    def __init__(self, p=0.33):
        super(FractionalDropout, self).__init__()
        self.p = p

    def forward(self, x):
        if not self.training:
            return x
        noise = torch.rand_like(x)
        mask = (noise > self.p).float()
        scale = 1.0 + (torch.rand_like(x) * 0.5)
        x = x * mask * scale
        return x
