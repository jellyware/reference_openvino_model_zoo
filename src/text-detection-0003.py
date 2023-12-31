import cv2
import numpy as np
 
# モジュール読み込み 
from openvino.inference_engine import IECore

# IEコアの初期化
ie = IECore()

#モデルの準備
file_path = 'intel/text-detection-0003/FP32/text-detection-0003'
model= file_path + '.xml'
weights = file_path + '.bin'

# モデルの読み込み
net = ie.read_network(model=model, weights=weights)
exec_net = ie.load_network(network=net, device_name='CPU')

# 入出力データのキー取得 
input_blob = next(iter(net.input_info))
iter = iter(net.outputs)
out_blob_link = next(iter)
out_blob_segm = next(iter)

# 入力画像読み込み 
frame = cv2.imread('image/text.jpg')

# 入力データフォーマットへ変換 
img = cv2.resize(frame, (1280, 768)) # サイズ変更 
img = img.transpose((2, 0, 1))      # HWC > CHW 
img = np.expand_dims(img, axis=0)   # 次元合せ

# 推論実行 
out = exec_net.infer({input_blob: img})

# 出力から必要なデータのみ取り出し 
out_link = out[out_blob_link] 
out_link = out_link.transpose((0,2,3,1))
# print(out_link.shape)

out_segm = out[out_blob_segm] 
out_segm = out_segm.transpose((0,2,3,1))
# print(out_segm.shape)

# for i j in zip(out_link,out_segm):

# # 検出されたすべての顔領域に対して１つずつ処理 
# for detection in Nout:
#     # conf値の取得 
#     confidence = float(detection[2])

#     # バウンディングボックス座標を入力画像のスケールに変換 
#     xmin = int(detection[3] * frame.shape[1])
#     ymin = int(detection[4] * frame.shape[0])
#     xmax = int(detection[5] * frame.shape[1])
#     ymax = int(detection[6] * frame.shape[0])

#     # conf値が0.5より大きい場合のみ感情推論とバウンディングボックス表示 
#     if confidence > 0.5:
#         cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color=(240, 180, 0), thickness=3)
 
#     # 画像表示 
#     cv2.imshow('frame', frame)
 
# # キーが押されたら終了 
# cv2.waitKey(0)
# cv2.destroyAllWindows()