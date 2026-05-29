import torch

checkpoint = torch.load(
    r"C:\data_unibas\pcrnet_checkpoints_test\pcrnet_epoch_2.pth",
    map_location="cpu"
)

print(checkpoint.keys())
print("epoch:", checkpoint["epoch"])
print("train_loss:", checkpoint["train_loss"])
print("val_loss:", checkpoint["val_loss"])
print("max_iteration:", checkpoint["max_iteration"])