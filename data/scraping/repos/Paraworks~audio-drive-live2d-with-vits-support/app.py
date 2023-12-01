import argparse
from text import text_to_sequence
import numpy as np
from scipy.io import wavfile
import torch
import json
import commons
import utils
import sys
import pathlib
from flask import Flask, request
import multiprocessing
import threading
import openai
import onnxruntime as ort
import time
from ppgan.apps.wav2lip_predictor import Wav2LipPredictor
import io
app = Flask(__name__)
mutex = threading.Lock()
def get_args():
    parser = argparse.ArgumentParser(description='inference')
    parser.add_argument('--audio',
                    type=str,
                    help='中途生成的语音存储路径',
                    default = '../res/audio.wav')
    parser.add_argument('--face',
                    type=str,
                    help='中之人的照片，用于talking face',
                    default = '../res/stdface.jpg')
    parser.add_argument('--outfile',
                    type=str,
                    help='视频路径，动作捕捉的替代品',
                    default='../res/result_voice.mp4')
    parser.add_argument('--texture',
                    type=str,
                    help='Texture file contain the response of chatbot.',
                    default='../res/status.txt')
    parser.add_argument('--waitVideo',
                    type=str,
                    help='Texture file contain the response of chatbot.',
                    default='../res/Masahiro.mp4')
    parser.add_argument('--onnx_model', 
                    type=str,
                    help = 'onnx checkpoint',
                    default = './moe/model.onnx')
    parser.add_argument('--cfg', 
                    type=str,
                    help = 'onnx config',
                    default="./moe/config.json")
    parser.add_argument('--outdir', 
                    type=str,
                    help='ouput directory',
                    default="./moe")
    parser.add_argument('--key',
                    type=str,
                    help='openai的key',
                    default = "your_key")
    args = parser.parse_args()
    return args

def to_numpy(tensor: torch.Tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad \
        else tensor.detach().numpy()

def get_symbols_from_json(path):
    import os
    assert os.path.isfile(path)
    with open(path, 'r') as f:
        data = json.load(f)
    return data['symbols']

args = get_args()
symbols = get_symbols_from_json(args.cfg)
phone_dict = {
        symbol: i for i, symbol in enumerate(symbols)
    }
hps = utils.get_hparams_from_file(args.cfg)
ort_sess = ort.InferenceSession(args.onnx_model, providers=['CUDAExecutionProvider'])
outdir = args.outdir
def is_japanese(string):
        for ch in string:
            if ord(ch) > 0x3040 and ord(ch) < 0x30FF:
                return True
        return False 

def gpt_chat(text):
  call_name = "派蒙"
  openai.api_key = args.key
  identity = "用中文回答我的问题"
  start_sequence = '\n'+str(call_name)+':'
  restart_sequence = "\nYou: "
  if 1 == 1:
     prompt0 = text #当期prompt
  if text == 'quit':
     return prompt0
  prompt = identity + prompt0 + start_sequence
  response = openai.Completion.create(
    model="text-davinci-003",
    prompt=prompt,
    temperature=0.5,
    max_tokens=1000,
    top_p=1.0,
    frequency_penalty=0.5,
    presence_penalty=0.0,
    stop=["\nYou:"]
  )
  return response['choices'][0]['text'].strip()

def ttv(text):
    sid = 16
    text = f"[JA]{text}[JA]" if is_japanese(text) else f"[ZH]{text}[ZH]"
    seq = text_to_sequence(text, cleaner_names=hps.data.text_cleaners
                                   )
    if hps.data.add_blank:
        seq = commons.intersperse(seq, 0)
    with torch.no_grad():
        x = np.array([seq], dtype=np.int64)
        x_len = np.array([x.shape[1]], dtype=np.int64)
        sid = np.array([sid], dtype=np.int64)
        scales = np.array([0.667, 0.8, 1], dtype=np.float32)
        scales.resize(1, 3)
        ort_inputs = {
                    'input': x,
                    'input_lengths': x_len,
                    'scales': scales,
                    'sid': sid
                }
        t1 = time.time()
        audio = np.squeeze(ort_sess.run(None, ort_inputs))
        audio *= 32767.0 / max(0.01, np.max(np.abs(audio))) * 0.6
        audio = np.clip(audio, -32767.0, 32767.0)
        bytes_wav = bytes()
        byte_io = io.BytesIO(bytes_wav)
        wavfile.write(args.audio,hps.data.sampling_rate, audio.astype(np.int16))
        t2 = time.time()
        spending_time = "推理时间："+str(t2-t1)+"s" 
        print(spending_time)
    predictor = Wav2LipPredictor(checkpoint_path=None,
                                 static=False,
                                 fps=25,
                                 pads=[0, 10, 0, 0],
                                 face_det_batch_size=16,
                                 wav2lip_batch_size=128,
                                 resize_factor=1,
                                 crop=[0, -1, 0, -1],
                                 box=[-1, -1, -1, -1],
                                 rotate=False,
                                 nosmooth=False,
                                 face_detector='sfd',
                                 face_enhancement=False)
    predictor.run(args.face, args.audio,args.outfile)
    with open(args.texture, "w", encoding="utf-8") as f1:
        text = f1.write('跳过这句话')


def main():
    while True:
        c = input("choice")
        if int(c) == 1:
            your_text = input("你：")
            t1 = time.time()
            text = gpt_chat(your_text)
            t2 = time.time()
            print("api回复你总共用了", (t2 - t1), "s，真是太棒啦！")
            print('回答：'+text)
            text = text.replace('\n','。').replace(' ',',')
            ttv(text)
        else:
            text = input("：")
            ttv(text)

if __name__ == '__main__':
    main()
