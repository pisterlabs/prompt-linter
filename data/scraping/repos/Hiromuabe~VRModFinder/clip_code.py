
import torch 
import clip
import PIL.Image
import objaverse
import cv2
import numpy as np
from IPython.display import Image, display
from matplotlib import pyplot as plt
import requests
import tempfile
from tkinter import *
import re
import openai
import os
import json
import objaverse
import multiprocessing as mp
from tqdm import tqdm
def imread_web(url):
    # 画像をリクエストする
    res = requests.get(url)
    img = None
    # Tempfileを作成して即読み込む
    with tempfile.NamedTemporaryFile(dir='./') as fp:
        fp.write(res.content)
        fp.file.seek(0)
        img = cv2.imread(fp.name)
    return img

def imshow(img):
    """ndarray 配列をインラインで Notebook 上に表示する。
    """
    ret, encoded = cv2.imencode(".jpg", img)
    display(Image(encoded))

def chenge_img(url):
    IMG_PATH = imread_web(url)
    img = cv2.cvtColor(IMG_PATH,cv2.COLOR_BGR2RGB)
    return img

def score_clip(url:str,feature:str,device="cpu"):
    # テキスト 
    TEXT_LIST = [feature,"not "+feature]
    # モデルの読み込み 
    model, preprocess = clip.load("ViT-B/32", device=device)
    result_percent = []
    try:
        img = chenge_img(url)
        image = preprocess(PIL.Image.fromarray(img)).unsqueeze(0).to(device)
    except:
        result_percent.append(None)
        return 0.0
    text = clip.tokenize(TEXT_LIST).to(device) 
    with torch.no_grad(): 
        # 画像、テキストのエンコード 
        image_features = model.encode_image(image) 
        text_features = model.encode_text(text) 
        # 推論 
        logits_per_image, logits_per_text = model(image, text) 
        probs = logits_per_image.softmax(dim=-1).cpu().numpy()
        return probs[0][0]
    
def squeeze(interior_names):
    print(f"squeeze:{interior_names}")
    file = open("data.json")
    file = json.load(file)
    interior_names = interior_names.split(",")
    names = [[] for _ in range(len(interior_names))]
    urls = [[] for _ in range(len(interior_names))]
    uids = [[] for _ in range(len(interior_names))]

    for _ in range(len(interior_names)):
        points = []
        interior = interior_names[_]
        print("インテリア名")
        print(interior)
        # interior = "writing desk"
        # interior_name_parts = ["writing","desk"]
        interior_name_parts = interior.split(" ")
        
        if len(interior_name_parts) == 1:
            for j,k in zip(file.keys(),file.values()):
                if interior_name_parts[0] in k["name"].split(" "):
                    names[_].append(k["name"])
                    urls[_].append(k["thumbnail_url"])
                    uids[_].append(j)
        else:
            for j,k in zip(file.keys(),file.values()):
                if interior in k["name"].lower():
                    names[_].append(k["name"])
                    urls[_].append(k["thumbnail_url"])
                    uids[_].append(j)
                all_point = 0
                name_parts = k["name"].lower().split(" ")
                for i in range(len(interior_name_parts)):
                    if interior_name_parts[i] in name_parts:
                        if i == len(interior_name_parts)-1:
                            all_point += 100
                        else:
                            all_point += i+1
                points.append(all_point)
            max_point = max(points)
            print(max_point)
            threshold = max_point
            sorted_index = [index for index, value in sorted(enumerate(points),key=lambda x:-x[1]) if value == threshold]
            uid_list = list(file.keys())
            for num in sorted_index:
                names[_].append(file[uid_list[num]]["name"])
                urls[_].append(file[uid_list[num]]["thumbnail_url"])
                uids[_].append(uid_list[num])
        names[_] = list(set(names[_]))
        urls[_] = list(set(urls[_]))
        uids[_] = list(set(uids[_]))
    return names, urls, uids

def get_uid(interior_list, features_list,names,urls,uids,device="cpu"):
    scores = [[] for _ in range(len(interior_list))]
    selected_uid = []
    selected_name = []
    for i in tqdm(range(len(urls))):
        for j in tqdm(range(len(urls[i]))):
            scores[i].append(score_clip(urls[i][j],features_list[i],device))
        index = scores[i].index(max(scores[i]))
        selected_uid.append(uids[i][index])
        selected_name.append(names[i][index])
    return selected_name, selected_uid

def install_obj(uid):
    objaverse.load_objects(
        uids=uid
    )
    print("finish")
    return