import torch
import torch.nn as nn
import torch.nn.functional as F
from .pooling import Pooling


class PointNet(torch.nn.Module):
    def __init__(self, emb_dims=1024, input_shape="bnc", input_dim=36):
        super(PointNet, self).__init__()

        if input_shape not in ["bcn", "bnc"]:
            raise ValueError("Allowed shapes are 'bcn' or 'bnc'")

        self.input_shape = input_shape
        self.emb_dims = emb_dims
        self.input_dim = input_dim
        self.layers = self.create_structure()

    def create_structure(self):
        self.conv1 = torch.nn.Conv1d(self.input_dim, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 64, 1)
        self.conv3 = torch.nn.Conv1d(64, 64, 1)
        self.conv4 = torch.nn.Conv1d(64, 128, 1)
        self.conv5 = torch.nn.Conv1d(128, self.emb_dims, 1)
        self.relu = torch.nn.ReLU()

        layers = [
            self.conv1, self.relu,
            self.conv2, self.relu,
            self.conv3, self.relu,
            self.conv4, self.relu,
            self.conv5, self.relu
        ]

        return layers

    def forward(self, input_data):
        if self.input_shape == "bnc":
            input_data = input_data.permute(0, 2, 1)

        if input_data.shape[1] != self.input_dim:
            raise RuntimeError(
                f"Expected input with {self.input_dim} channels, "
                f"but got {input_data.shape[1]}"
            )

        output = input_data

        for layer in self.layers:
            output = layer(output)

        return output


if __name__ == "__main__":
    x = torch.rand((10, 1024, 36))

    pn = PointNet(input_dim=36)
    y = pn(x)

    print("Network Architecture:")
    print(pn)
    print("Input Shape:", x.shape)
    print("Output Shape:", y.shape)