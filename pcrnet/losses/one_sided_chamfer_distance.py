import torch
import torch.nn as nn


def one_sided_chamfer_distance(template, source):
    """
    One-sided Chamfer distance: source -> template.

    template: [B, N_template, 3]  full target femur
    source:   [B, N_source, 3]    transformed partial source

    Measures how close each source point is to the target surface.
    """

    pairwise_dist = (source.unsqueeze(2) - template.unsqueeze(1)).pow(2).sum(dim=3)

    min_source_to_template = pairwise_dist.min(dim=2)[0]

    loss = min_source_to_template.mean()

    return loss


class OneSidedChamferDistanceLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, template, source):
        return one_sided_chamfer_distance(template, source)