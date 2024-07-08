import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, input_dim=371712, hidden_dims=[256, 512, 256], output_dim=2):
        super(MLP, self).__init__()

        # モデルの構築
        self.layers = nn.ModuleList()

        # 入力層
        self.layers.append(nn.Linear(input_dim, hidden_dims[0]))
        self.layers.append(nn.ReLU())

        # 隠れ層 (中間層)
        for i in range(len(hidden_dims) - 1):
            self.layers.append(nn.Linear(hidden_dims[i], hidden_dims[i + 1]))
            self.layers.append(nn.ReLU())

        # 出力層
        self.layers.append(nn.Linear(hidden_dims[-1], output_dim))

    def forward(self, x):
        x = x.view(x.size(0), -1)  # 入力をフラット化
        for layer in self.layers:
            x = layer(x)
        return x
        # print("フラット化された入力の形状:", x.shape)  # デバッグ用の行

        # for idx, layer in enumerate(self.layers):
        #     # if idx == 0:  # これは最初の線形層です
        #     #     # print("最初の線形層の重みの形状:", layer.weight.shape)  # デバッグ用の行
        #     # x = layer(x)
        # return x


# # 使用例
# input_dim = 123904
# hidden_dims = [512, 256]
# output_dim = 2

# model = MLP(input_dim, hidden_dims, output_dim)
# print(model)
