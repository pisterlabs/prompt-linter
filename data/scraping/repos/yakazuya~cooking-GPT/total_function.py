import numpy as np
from ultralytics import YOLO
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
import time
import openai
import re

import torch
from diffusers import StableDiffusionPipeline

from modules.api_key import API_KEY,ORGANIZATION_KEY

def gpt(food_list:list) -> str:
    openai.organization = ORGANIZATION_KEY
    openai.api_key = API_KEY


    if len(food_list) == 1:
        connect_text:str = f'Please tell me what dishes can be made using only {food_list[-1]}.' \
                                'Please display in the order of: number, dish name, colon, space, and description. And please do not include line breaks between the sentences.'
    else:
        connect_text:str = f'Please tell me what dishes can be made using only' \
                                + ', '.join(food_list[:len(food_list)-1])  \
                                        + f' and {food_list[-1]}.' \
                                            + 'Please display in the order of: number, dish name, colon, space, and description. And please do not include line breaks between the sentences.'
    
    
    response = openai.Completion.create(
        engine='text-davinci-003',
        prompt=connect_text,
        max_tokens=2000,
        temperature=0.7,
        n=1,
        stop=None,
    )
    
    return response.choices[0].text



def yolo(img_path:str) -> list:
    model = YOLO('./model/best.pt')
    results = model(img_path,save=True,conf=0.2, iou=0.5)
    names = results[0].names
    classes = results[0].boxes.cls
    names= [ names[int(cls)] for cls in classes] 

    vegetables_list=[]
    for value in names.values():
        vegetables_list.append(value)
    return vegetables_list



def translate(text:str, from_lang:str, to_lang:str) -> str:
    """
    引数    : 翻訳したいテキスト, 翻訳したいテキストの言語, 翻訳する言語
    返り値  : 翻訳されたテキスト
    """
    # seleniumとchoromDriverでDeepLにアクセス
    load_url = 'https://www.deepl.com/translator#' + from_lang +'/' + to_lang + '/' + text
    chrome_path = '/opt/google/chrome/google-chrome'

    options = Options()
    options.binary_location = chrome_path
    options.headless = True
    options.add_argument('--no-sandbox')
    options.add_argument('--disable-dev-shm-usage')
    driver = webdriver.Chrome(executable_path='/usr/local/bin/chromedriver', options=options)  #  driver = webdriver.Chrome()
    driver.get(load_url)

    time.sleep(5)

    #2023/04/21
    # output = ".lmt__textarea_container .lmt__inner_textarea_container d-textarea"
    output_selector = 'd-textarea.lmt__textarea.lmt__target_textarea.lmt__textarea_base_style.focus-visible-disabled-container'
    Outputtext = driver.find_element_by_css_selector(output_selector).get_attribute("textContent")
    
    return Outputtext

def dishes_select(response,sug) -> list:
    '''GPTからの料理名を受け取るリストの作成'''
    
    pattern = re.compile(pattern=r'\d+.\s*(.*?):')
    
    text_list:list = pattern.findall(response)
    
    recipe_list :list = list()
    for match in text_list:
        recipe_name = match.strip()
        recipe_list.append(recipe_name)
        
    text = " ".join(response.split("\n"))
    
    description_list = list()
    for recipe in recipe_list[:sug]:
        pattern = re.compile(pattern=fr'{recipe}\:\s(.+?)\d+\.')
        description = pattern.findall(text)
        description_list.append(description[0])
        
        
    if sug > len(recipe_list):
        sug = len(recipe_list)
    
    return recipe_list[:sug],description_list

def image(dish_list: list,text_list:list) -> list:
    # PIL形式
    model_id = "CompVis/stable-diffusion-v1-4"
    device = "cuda"
    # インスタンス化とcudaに転送
    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
    pipe = pipe.to(device)
    img_list = []
    for i, dish in enumerate(dish_list):
        prompt = "a photo of " + dish +f' is made by {text_list[i]}'
        # 多分画像生成してるところ
        image = pipe(prompt).images[0]  
        image.save(f"./static/images/{i}.png")
        img_list.append(f'static/images/{i}.png')
    
    return img_list

def molding(dish_translate, discription_translate, img_list):
    dict = {}
    for key, discription, img in zip(dish_translate, discription_translate, img_list):
        dict[key]={}
        dict[key]['description'] = discription
        dict[key]['img_url'] = img
    
    return dict
