import cv2
import numpy as np
 
# モジュール読み込み 
from openvino.inference_engine import IECore

# IEコアの初期化
ie = IECore()

#モデルの準備
file_path = 'intel/converted_model/saved_model'
model= file_path + '.xml'
weights = file_path + '.bin'

# モデルの読み込み
net = ie.read_network(model=model, weights=weights)
exec_net = ie.load_network(network=net, device_name='CPU')

# 入出力データのキー取得 
input_blob = next(iter(net.input_info))
out_blob = next(iter(net.outputs))

# 入力画像読み込み 
frame = cv2.imread('image/people.jpg')

# 入力データフォーマットへ変換 
img = cv2.resize(frame, (224, 224)) # サイズ変更 
img = img.transpose((2, 0, 1))      # HWC > CHW 
img = np.expand_dims(img, axis=0)   # 次元合せ

# 推論実行 
out = exec_net.infer({input_blob: img})

# 出力から必要なデータのみ取り出し 
out = out[out_blob] 
out = np.squeeze(out) #サイズ1の次元を全て削除 

#ラベル準備
f = open('intel/converted_model/labels.txt', 'r')
chars = [class_label.strip("\n") for class_label in f.readlines()]

# 出力値が最大のインデックスを得る 
index_max = np.argmax(out)

# 文字列描画
cv2.putText(frame, chars[index_max], (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

# 画像表示 
cv2.imshow('image', frame)

# 終了処理 
cv2.waitKey(0)
cv2.destroyAllWindows()