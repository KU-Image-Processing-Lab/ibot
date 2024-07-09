import numpy as np
import cv2

# 画像をロードし、NumPy配列に変換
image = cv2.imread('./dataset/sekkai_TrainDataset/images/left_2020_030690.png')
image = image[:, :, 2]

# NumPy配列の形状を確認
print(image.shape)  # (height, width, channels) の形でサイズを表示

# # 各チャンネルを分割
# b, g, r = cv2.split(image)
# b=str(b)
# g=str(g)
# r=str(r)
# print('b'+b,'g'+g,'r'+r)

# # チャンネルが等しいかどうかを確認
# if np.array_equal(b, g) and np.array_equal(g, r):
#     print("すべてのチャンネルが等しいです。")
# else:
#     print("チャンネルに違いがあります。")
