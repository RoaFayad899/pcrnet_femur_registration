import torch
import torch.nn as nn
import torch.nn.functional as F


# ==========================================================
# GEODESIC ROTATION LOSS
# ==========================================================

def geodesic_rotation_loss(R_pred, R_gt):
    """
    Geodesic distance between rotation matrices.

    R_pred: [B, 3, 3]
    R_gt:   [B, 3, 3]

    Returns:
        mean geodesic distance in radians
    """

    R_diff = torch.bmm(
        R_pred.transpose(1, 2),
        R_gt
    )

    trace = (
        R_diff[:, 0, 0]
        + R_diff[:, 1, 1]
        + R_diff[:, 2, 2]
    )

    cos_theta = (trace - 1.0) / 2.0

    # numerical stability
    cos_theta = torch.clamp(
        cos_theta,
        min=-1.0 + 1e-7,
        max=1.0 - 1e-7
    )

    theta = torch.acos(cos_theta)

    return theta.mean()


# ==========================================================
# TRANSLATION MSE LOSS
# ==========================================================

def translation_mse_loss(t_pred, t_gt):
    """
    Translation MSE.

    t_pred: [B, 1, 3] or [B, 3]
    t_gt:   [B, 3]
    """

    if t_pred.ndim == 3:
        t_pred = t_pred.squeeze(1)

    return F.mse_loss(t_pred, t_gt)


# ==========================================================
# TOTAL LOSS
# ==========================================================

class GeodesicTranslationLoss(nn.Module):

    def __init__(self, lambda_translation=1.0):
        super().__init__()

        self.lambda_translation = lambda_translation

    def forward(
        self,
        R_pred,
        t_pred,
        R_gt,
        t_gt
    ):

        rotation_loss = geodesic_rotation_loss(
            R_pred,
            R_gt
        )

        translation_loss = translation_mse_loss(
            t_pred,
            t_gt
        )

        total_loss = (
            rotation_loss
            + self.lambda_translation * translation_loss
        )

        return {
            "total_loss": total_loss,
            "rotation_loss": rotation_loss,
            "translation_loss": translation_loss
        }