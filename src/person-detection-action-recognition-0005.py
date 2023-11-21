import cv2
import numpy as np
 
# モジュール読み込み 
from openvino.inference_engine import IECore

# IEコアの初期化
ie = IECore()

#モデルの準備
file_path = 'intel/person-detection-action-recognition-0005/FP32/person-detection-action-recognition-0005'
model= file_path + '.xml'
weights = file_path + '.bin'

# モデルの読み込み
net = ie.read_network(model=model, weights=weights)
exec_net = ie.load_network(network=net, device_name='CPU')

# 入出力データのキー取得 
input_blob = next(iter(net.input_info))
iter = iter(net.outputs)

out_blob_prior_box = next(iter)
out_blob_box = next(iter)
out_blob_conf = next(iter)
out_blob_anchor1 = next(iter)
out_blob_anchor2 = next(iter)
out_blob_anchor3 = next(iter)
out_blob_anchor4 = next(iter)

# 入力画像読み込み 
frame = cv2.imread('image/people.jpg')

# 入力データフォーマットへ変換 
img = cv2.resize(frame, (680, 400)) # サイズ変更 
img = img.transpose((2, 0, 1))      # HWC > CHW 
img = np.expand_dims(img, axis=0)   # 次元合せ

# 推論実行 
out = exec_net.infer({input_blob: img})

# 出力から必要なデータのみ取り出し 
out_prior_box = out[out_blob_prior_box] 
out_prior_box = np.squeeze(out) #サイズ1の次元を全て削除 

out_box = out[out_blob_box] 
out_box = np.squeeze(out_box) #サイズ1の次元を全て削除 

out_conf = out[out_blob_conf]
out_conf = np.squeeze(out_conf) #サイズ1の次元を全て削除 

out_anchor1 = out[out_blob_anchor1] 
out_anchor1 = np.squeeze(out_anchor1) #サイズ1の次元を全て削除 

out_anchor2 = out[out_blob_anchor2] 
out_anchor2 = np.squeeze(out_anchor2) #サイズ1の次元を全て削除 

out_anchor3 = out[out_blob_anchor3] 
out_anchor3 = np.squeeze(out_anchor3) #サイズ1の次元を全て削除 

out_anchor4 = out[out_blob_anchor4] 
out_anchor4 = np.squeeze(out_anchor4) #サイズ1の次元を全て削除 

actions = ["sitting", "standing", "raising hand"]

num_priors = len(out_box)//4
# Box coordinates in SSD format (priot_boxからの差異)
out_boxs = out_box.reshape(num_priors,4)
out_confs = out_conf.reshape(num_priors,2)
# out_prior_boxes = out_prior_box.reshape(num_priors,4)

print(out_prior_box.shape)

# coordinates = []
# for out_box, out_conf, out_prior_box in zip(out_boxs, out_confs, out_prior_boxes):
#     x, y, w, h = out_box
#     coordinate = [out_prior_box[0]+x, out_prior_box[1]+y, out_prior_box[2]+w, out_prior_box[3]+h]
#     coordinates.append(coordinate)
#     conf1, conf2 = out_conf

# print(out_confs[0])

    # バウンディングボックス座標を入力画像のスケールに変換 
    # xmin = int(x * frame.shape[1])
    # ymin = int(y * frame.shape[0])
    # xmax = int((x + w) * frame.shape[1])
    # ymax = int((y + h) * frame.shape[0])



# Detection confidences (softmax)
# conf1 + conf2 = 1.0


# out_anchor1 = out_anchor1.reshape(num_priors,3)
# print(out_anchor1)
# print(anchor1)
# top_indices = [i for i, conf in enumerate(det_conf) if conf >= 0.6]


# for i in range(0,len(out_conf), 2):
#     conf1, conf2 = out_conf[i: i+2]



# for detection in actionOut[2].reshape(-1, 3):
#     print('sitting ' +str( detection[0]))
#     print('standing ' +str(detection[1]))
#     print('raising hand ' +str(detection[2]))


# 検出されたすべての顔領域に対して１つずつ処理 
# for detection in out:
#     # conf値の取得 
#     confidence = float(detection[2])

    # バウンディングボックス座標を入力画像のスケールに変換 
    # xmin = int(out_box[0] * frame.shape[1])
    # ymin = int(out_box[1] * frame.shape[0])
    # xmax = int(out_box[2] * frame.shape[1])
    # ymax = int(out_box[3] * frame.shape[0])


#     # conf値が0.5より大きい場合のみ感情推論とバウンディングボックス表示 
#     if confidence > 0.5:
    # cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color=(240, 180, 0), thickness=3)
 
# 画像表示 
# cv2.imshow('frame', frame)
 
# キーが押されたら終了 
cv2.waitKey(0)
cv2.destroyAllWindows()
