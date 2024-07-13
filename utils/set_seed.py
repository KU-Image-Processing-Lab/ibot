import os
import random

import numpy as np
import torch


def seed_everything(seed=42):
    random.seed(seed)  # Pythonの組み込み乱数生成器を固定
    os.environ["PYTHONHASHSEED"] = str(seed)  # Pythonのハッシュ乱数生成器を固定
    np.random.seed(seed)  # Numpyの乱数生成器を固定
    torch.manual_seed(seed)  # CPUでの乱数生成器を固定
    torch.cuda.manual_seed(seed)  # GPUでの乱数生成器を固定
    torch.cuda.manual_seed_all(seed)  # マルチGPU環境でも乱数生成器を固定
    torch.backends.cudnn.deterministic = (
        True  # CuDNNを使用する場合の振る舞いを決定論的に
    )
    torch.backends.cudnn.benchmark = False  # ネットワークのアーキテクチャが変わらない場合はTrueの方が高速化されるが、再現性を確保するためFalseに設定
