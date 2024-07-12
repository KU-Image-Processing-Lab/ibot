import torch


# モデルの定義
class ACC_UNet(nn.Module):
    def __init__(self):
        super(ACC_UNet, self).__init__()
        # モデルの定義 (例)
        self.fc = nn.Linear(10, 1)

    def forward(self, x):
        return self.fc(x)


# モデルのインスタンスを作成
model = ACC_UNet()
optimizer = optim.Adam(model.parameters())

# 元のチェックポイントファイルのパス
model_path_tar = "./snapshots/ACCUnet/e-4/e-4Transfuse-49.pth.tar"

# チェックポイントの読み込み
checkpoint = torch.load(model_path_tar)

# 'model_state_dict' キーが存在するか確認
if "model_state_dict" in checkpoint:
    # 新しいチェックポイントを作成
    new_checkpoint = checkpoint["model_state_dict"]
else:
    # パラメータのみが保存されたチェックポイントの場合
    new_checkpoint = checkpoint

# 新しいチェックポイントファイルのパス
model_path_pth = "./snapshots/ACCUnet/e-4/e-4Transfuse-49.pth"

# モデルの状態を新しいファイルに保存
torch.save(new_checkpoint, model_path_pth)

print(f"モデルの状態が {model_path_tar} から {model_path_pth} に保存されました。")
