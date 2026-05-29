import torch
import torch.nn as nn
from .cuda.emd import EMDLoss


def emd(template: torch.Tensor, source: torch.Tensor):
    emd_module = EMDLoss()
    emd_loss = torch.mean(emd_module(template, source)) / template.size(1)
    return emd_loss


class EMDLossWrapper(nn.Module):
    def __init__(self):
        super(EMDLossWrapper, self).__init__()

    def forward(self, template, source):
        return emd(template, source)