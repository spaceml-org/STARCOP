import torch.nn.functional as F

def l1(input, target):
    return F.l1_loss(input, target)

def mse(input, target):
    return F.mse_loss(input, target)
