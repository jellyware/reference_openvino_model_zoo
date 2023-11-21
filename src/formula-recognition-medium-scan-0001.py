from calendar import c
import cv2
import numpy as np
 
# モジュール読み込み 
from openvino.inference_engine import IECore

# IEコアの初期化
ie = IECore()

#モデルの準備(Encoder)
file_path_en = 'intel/formula-recognition-medium-scan-0001/formula-recognition-medium-scan-0001-im2latex-encoder/FP32/formula-recognition-medium-scan-0001-im2latex-encoder'
model_en= file_path_en + '.xml'
weights_en = file_path_en + '.bin'

#モデルの準備(Decoder)
file_path_de = 'intel/formula-recognition-medium-scan-0001/formula-recognition-medium-scan-0001-im2latex-decoder/FP32/formula-recognition-medium-scan-0001-im2latex-decoder'
model_de= file_path_de + '.xml'
weights_de = file_path_de + '.bin'

# モデルの読み込み(Encoder)
net_en = ie.read_network(model=model_en, weights=weights_en)
exec_net_en = ie.load_network(network=net_en, device_name='CPU')

# モデルの読み込み(Decoder)
net_de = ie.read_network(model=model_de, weights=weights_de)
exec_net_de = ie.load_network(network=net_de, device_name='CPU')

# 入出力データのキー取得 (Encoder)
input_blob_en = next(iter(net_en.input_info))
iter = iter(net_en.outputs)
out_blob_en_con = next(iter)
out_blob_en_hid = next(iter)
out_blob_en_init = next(iter)
out_blob_en_row = next(iter)

# カメラ準備 
frame = cv2.imread("image/formula.jpg")

# 入力データフォーマットへ変換 (Encoder)
img = cv2.resize(frame, (1400, 160)) # サイズ変更 
img = img.transpose((2, 0, 1))      # HWC > CHW 
img = np.expand_dims(img, axis=0)   # 次元合せ

# 推論実行 (Encoder)
out_en = exec_net_en.infer({input_blob_en: img})

# 出力から必要なデータのみ取り出し (Encoder)
out_en_con = out_en[out_blob_en_con] 
out_en_hid = out_en[out_blob_en_hid] 
out_en_init = out_en[out_blob_en_init] 
out_en_row = out_en[out_blob_en_row] 

# 入出力データのキー取得 (Decoder)
input_blob_de_c = net_de.input_info["dec_st_c"]
input_blob_de_h = net_de.input_info["dec_st_h"]
input_blob_de_prev = net_de.input_info["output_prev"]
input_blob_de_row = net_de.input_info["row_enc_out"]
input_blob_de_tgt = net_de.input_info["tgt"]

# 推論実行 (Decoder)
out_de_c = exec_net_de.infer({input_blob_de_c: out_en_con})
out_de_h = exec_net_de.infer({input_blob_de_h: out_en_hid})
out_de_prev = exec_net_de.infer({input_blob_de_prev: out_en_init})
out_de_row = exec_net_de.infer({input_blob_de_row: out_en_row})

out_de_prev = exec_net_de.infer({"output_prev": out_en_init})
out_de_row = exec_net_de.infer({"row_enc_out": out_en_row})

print(out_de_prev)

# out_de_tgt = exec_net_de.infer({: decoder_input})

lstm_context = out_de_row["dec_st_c_t"]
lstm_hidden = out_de_row["dec_st_h_t"]

# print(lstm_context.shape)
# print(lstm_hidden.shape)

output = out_de_row["output"]
confidence = out_de_row["logit"]


model = Model(interactive_mode)

# phrase = model.vocab.construct_phrase(targets)

def calculate_probability(distribution):
    return np.prod(np.amax(distribution, axis=1))


from tqdm import tqdm

def non_interactive_demo(model, args):
    renderer = create_renderer()
    show_window = not args.no_show
    for rec in tqdm(model.images_list):
        image = rec.img
        distribution, targets = model.infer_sync(image)
        prob = calculate_probability(distribution)
        if prob >= args.conf_thresh ** len(distribution):
            phrase = model.vocab.construct_phrase(targets)
            if args.output_file:
                with open(args.output_file, 'a') as output_file:
                    output_file.write(rec.img_name + '\t' + phrase + '\n')
            else:
                print("\n\tImage name: {}\n\tFormula: {}\n".format(rec.img_name, phrase))
                if renderer is not None:
                    rendered_formula, _ = renderer.render(phrase)
                    if rendered_formula is not None and show_window:
                        cv2.imshow("Predicted formula", rendered_formula)
                        cv2.waitKey(0)

DEFAULT_WIDTH = 800


def _create_input_window(self):
    aspect_ratio = self._tgt_shape[0] / self._tgt_shape[1]
    width = min(DEFAULT_WIDTH, self.resolution[0])
    height = int(width * aspect_ratio)
    start_point = (int(self.resolution[0] / 2 - width / 2), int(self.resolution[1] / 2 - height / 2))
    end_point = (int(self.resolution[0] / 2 + width / 2), int(self.resolution[1] / 2 + height / 2))
    return start_point, end_point

start_point, end_point = _create_input_window()
_latex_h = 0

import re

def strip_internal_spaces(text):
    text = text.replace("{ ", "{")
    text = text.replace(" }", "}")
    text = text.replace("( ", "(")
    text = text.replace(" )", ")")
    text = text.replace(" ^ ", "^")
    return re.sub(r'(?<=[\d.]) (?=[\d.])', '', text)

def _put_text(frame, text):
    # if text == '':
    #     return frame
    text = strip_internal_spaces(text)
    (txt_w, _latex_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, 3)
    start_point = (start_point[0], end_point[1] - start_point[1] + int(_latex_h * 1.5))
    comment_coords = (0, end_point[1] - start_point[1] + int(_latex_h * 1.5))
    frame = cv2.putText(frame, "Rendered:", comment_coords, fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.7, color=[0,0,0], thickness=2, lineType=cv2.LINE_AA)
    return frame

import subprocess

def create_renderer():
    command = subprocess.run(["pdflatex", "--version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
    if command.returncode != 0:
        renderer = None
    else:
        renderer = Renderer()
    return renderer

_renderer = create_renderer()

cur_formula = None
res_img = None
_state = Renderer.Status.READY
_worker = ThreadPool(processes=1)
_async_result = None

import os
import tempfile
import sympy
DENSITY = 300


def render(formula):
    """
    Synchronous method. Returns rendered image and text of the formula,
    corresponding to the rendered image when rendering is done.
    """
    if cur_formula is None:
        cur_formula = formula
    elif cur_formula == formula:
        return res_img, cur_formula
    cur_formula = formula
    res_img = None

    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as output_file:
        output_file_name = output_file.name

    try:
        sympy.preview('$${}$$'.format(formula), viewer='file',
                        filename=output_file_name, euler=False, dvioptions=['-D', '{}'.format(DENSITY)])
        res_img = cv2.imread(output_file_name)
    finally:
        os.unlink(output_file_name)

    return res_img, cur_formula


from multiprocessing.pool import ThreadPool

_state = Renderer.Status.READY
_async_result = None
_worker = ThreadPool(processes=1)


def thread_render(formula):
    """
    Provides asynchronous interface to the rendering process.
    In contrast with the .render, this one returns image and formula only when rendering is complete.
    If rendering is incomplete at the moment of method call, method returns None
    """
    if _state == Renderer.Status.READY:
        _async_result = _worker.apply_async(render, args=(formula,))
        _state = Renderer.Status.RENDERING
        return None

    if _state == Renderer.Status.RENDERING:
        if _async_result.ready() and _async_result.successful():
            _state = Renderer.Status.READY
            return res_img, cur_formula
        elif _async_result.ready() and not _async_result.successful():
            _state = Renderer.Status.READY
        return None

def _render_formula_async( formula):
    if formula == _prev_rendered_formula:
        return _prev_formula_img
    result = _renderer.thread_render(formula) #★
    formula_img, res_formula = result
    if res_formula != formula:
        return None
    _prev_rendered_formula = formula
    _prev_formula_img = formula_img
    return formula_img

def _put_formula_img(frame, formula):
    formula_img = _render_formula_async(formula) #★

    y_start = end_point[1] - start_point[1] + _latex_h * 2
    frame[y_start:y_start + formula_img.shape[0], start_point[0]:start_point[0] + formula_img.shape[1],:] = formula_img
    comment_coords = (0, y_start + (formula_img.shape[0] + _latex_h) // 2) 
    frame = cv2.putText(frame, "Rendered:", comment_coords, fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.7, color=[0,0,0], thickness=2, lineType=cv2.LINE_AA)
    return frame


def draw(frame, phrase):
    frame = _put_text(frame, phrase)
    frame = _put_formula_img(frame, phrase)#★
    return frame

frame = draw(frame, phrase)


# print(out_de_row)

# 出力から必要なデータのみ取り出し (Decoder)
# out_blob_de = next(iter(net_de.outputs))
# print(net_de.outputs)
# out_de_c = out_de_c["dec_st_c_t"]
# out_de_h = out_de_h["dec_st_h_t"]
# out_de_prev = out_de_prev["output"]
# out_de_c = out_de_c["logit"]


# # #ラベル準備
# # f = open('others/kinetics_400.txt', 'r')
# # class_labels = f.readlines()

# # # encoder_output = []
# # # sample_duration = 16
# # # text = ""


# # if len(encoder_output) == sample_duration:

# #     # 入力データフォーマットへ変換 (Decoder)
# #     decoder_input = np.concatenate(encoder_output, axis=0)
# #     decoder_input = decoder_input.transpose((2, 0, 1, 3))
# #     decoder_input = np.squeeze(decoder_input, axis=3)


#     # 出力から必要なデータのみ取り出し 
#     out_de = out_de[out_blob_de]

#     # softmax表現
#     exp = np.exp(out_de - np.max(out_de))
#     probs = exp / np.sum(exp, axis=None)

#     # encorderのリストを空にする
#     encoder_output = []

#     # 出力値が最大のインデックスを得る 
#     index_max = np.argmax(probs)
#     probs = '{:.2f}'.format(float(np.amax(np.squeeze(probs))) * 100) 
#     text = class_labels[index_max].replace("\n", "") + "-" + str(probs) + "%"

# print(text)

# # # 文字列描画
# # cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

# # cv2.imshow('frame', frame)

# # # キーが押されたら終了 
# cv2.waitKey(0)
# cv2.destroyAllWindows()