import argparse

from lib.Trans_CaraNet import Trans_CaraNet_L

parser = argparse.ArgumentParser()
parser.add_argument("--testsize", type=int, default=352, help="testing size")
parser.add_argument(
    "--pth_path", type=str, default="./snapshots/Transfuse_S/Transfuse-99.pth"
)


parser.add_argument(
    "--pth_path2", type=str, default="./snapshots/Transfuse_S/Discriminator-99.pth"
)

parser.add_argument(
    "--discriminator",
    type=str,
    default="resnet10",
    help="Discriminatorに使うモデル",
)
parser.add_argument("--test_data_path1", type=str, default="./dataset/TestDataset/")
parser.add_argument(
    "--test_data_path2", type=str, default="./dataset/sekkai_TestDataset/"
)
parser.add_argument("--test_save_path", type=str, default="./results/Transfuse_S/")


opt = parser.parse_args()

model = Trans_CaraNet_L()

# ---- build models ----
if opt.discriminator == "resnet10":
    from lib.ResNet_10 import ResNet10

    model2 = ResNet10()

if opt.discriminator == "res2net":
    from lib.Discriminator_ResNet import Discriminator

    model2 = Discriminator()

from torchinfo import summary
summary(model,input_size=(16, 3, 352, 352),col_names=["output_size", "num_params"],)
summary(model2,input_size=(16, 3, 352, 352),col_names=["output_size", "num_params"],)
