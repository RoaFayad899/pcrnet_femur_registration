import torch
import torch.nn as nn
import torch.nn.functional as F
from .pointnet_fpfh import PointNet
from .pooling import Pooling
from ..ops.transform_functions import PCRNetTransform as transform


class iPCRNet(nn.Module):
    def __init__(self, feature_model=PointNet(), droput=0.0, pooling='max'):
        super().__init__()
        self.feature_model = feature_model
        self.pooling = Pooling(pooling)

        self.linear = [nn.Linear(self.feature_model.emb_dims * 2, 1024), nn.ReLU(),
                       nn.Linear(1024, 1024), nn.ReLU(),
                       nn.Linear(1024, 512), nn.ReLU(),
                       nn.Linear(512, 512), nn.ReLU(),
                       nn.Linear(512, 256), nn.ReLU()]

        if droput > 0.0:
            self.linear.append(nn.Dropout(droput))
        self.linear.append(nn.Linear(256, 9))

        self.linear = nn.Sequential(*self.linear)

    # Single Pass Alignment Module (SPAM)
    def spam(self, template_features, source, est_R, est_t):
        batch_size = source.size(0)

        self.source_features = self.pooling(self.feature_model(source))
        y = torch.cat([template_features, self.source_features], dim=1)
        pose_9d = self.linear(y)

        rot6d = pose_9d[:, 0:6]
        trans = pose_9d[:, 6:9]

        est_R_temp = self.rotation_6d_to_matrix(rot6d)
        est_t_temp = trans.view(-1, 1, 3)

        # update translation matrix.
        est_t = torch.bmm(est_R_temp, est_t.permute(0, 2, 1)).permute(0, 2, 1) + est_t_temp
        # update rotation matrix.
        est_R = torch.bmm(est_R_temp, est_R)

        # Split xyz and FPFH
        source_xyz = source[:, :, :3]
        source_fpfh = source[:, :, 3:]

        # Transform only xyz
        source_xyz = (
                torch.bmm(
                    source_xyz,
                    est_R_temp.transpose(1, 2)
                )
                + est_t_temp
        )

        # Reassemble
        source = torch.cat(
            [source_xyz, source_fpfh],
            dim=2
        )
        return est_R, est_t, source

    def forward(self, template, source, max_iteration=8):
        est_R = torch.eye(3).to(template).view(1, 3, 3).expand(
            template.size(0), 3, 3
        ).contiguous()

        est_t = torch.zeros(1, 3).to(template).view(1, 1, 3).expand(
            template.size(0), 1, 3
        ).contiguous()

        template_features = self.pooling(self.feature_model(template))

        intermediate_Rs = []
        intermediate_ts = []
        intermediate_sources = []

        for i in range(max_iteration):
            est_R, est_t, source = self.spam(
                template_features,
                source,
                est_R,
                est_t
            )

            intermediate_Rs.append(est_R)
            intermediate_ts.append(est_t)
            intermediate_sources.append(source)

        result = {
            'est_R': est_R,
            'est_t': est_t,
            'est_T': transform.convert2transformation(est_R, est_t),
            'r': template_features - self.source_features,
            'transformed_source': source,

            # Added for progressive loss and debugging
            'intermediate_Rs': intermediate_Rs,
            'intermediate_ts': intermediate_ts,
            'intermediate_sources': intermediate_sources,
        }

        return result

    def rotation_6d_to_matrix(self, rot6d):

        a1 = rot6d[:, 0:3]
        a2 = rot6d[:, 3:6]

        b1 = F.normalize(a1, dim=1, eps=1e-6)

        dot = torch.sum(b1 * a2, dim=1, keepdim=True)
        a2_orthogonal = a2 - dot * b1

        b2 = F.normalize(a2_orthogonal, dim=1, eps=1e-6)

        b3 = torch.cross(b1, b2, dim=1)

        R = torch.stack([b1, b2, b3], dim=2)

        return R


if __name__ == '__main__':
    template = torch.rand(10, 1024, 36)
    source = torch.rand(10, 1024, 36)
    pn = PointNet()

    net = iPCRNet(pn)
    result = net(template, source)
