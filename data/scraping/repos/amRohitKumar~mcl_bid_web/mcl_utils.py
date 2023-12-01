# -*- coding: utf-8 -*-
"""
Created on Tue Mar 28 23:31:32 2023

@author: mukhe
"""

import fitz
from PIL import Image
import pytesseract
import re
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import requests
import openai
import io
from dotenv import load_dotenv
import time
import easyocr  # for optical character recognition

load_dotenv()
API_KEY = os.getenv('API_KEY')

openai.api_key = API_KEY

# change when deploying
pytesseract.pytesseract.tesseract_cmd = r'/usr/bin/tesseract'

def open_file_link(filepath):
    filestream = io.BytesIO(requests.get(filepath).content)
    file = fitz.open(stream=filestream, filetype="pdf")
    filename = filepath.split("//")[-1].split(".")[0]
    fileloc = "\\".join(filepath.split("\\")[:-1])
    return file, filename, fileloc

def pdf2img(filepath):  
    pdf_file, filename, fileloc = open_file_link(filepath)
    images_extracted = []
    # iterate over PDF pages
    for page_index in range(len(pdf_file)):
        # get the page itself
        page = pdf_file[page_index]
        image_list = page.get_images()
        # printing number of images found in this page
        if image_list:
            print(f"[+] Found a total of {len(image_list)} images in page {page_index}")
        else:
            print("[!] No images found on page", page_index)
        for image_index, img in enumerate(page.get_images(), start=1):
            # get the XREF of the image
            xref = img[0]
            # extract the image bytes
            base_image = pdf_file.extract_image(xref)
            image_bytes = base_image["image"]
            # get the image extension
            image_ext = base_image["ext"]
            # load it to PIL
            image = Image.open(io.BytesIO(image_bytes))
            # save it to local disk
            image_name = f"{filename}P{page_index}I{image_index}.{image_ext}"
            image.save(open(image_name, "wb"))
            images_extracted.append(image_name)
    return images_extracted

def extract_text(filepath, startpg=1, endpg=None):
    file, filename, fileloc = open_file_link(filepath)
    if endpg == None:
        endpg = len(file)
    text = ""
    images = []
    for i in range(startpg-1, endpg):
        pix = file[i].get_pixmap()
        imgloc = fileloc + "\\" + filename + str(i) + ".jpg"
        images.append(imgloc)
        pix.save(imgloc, "JPEG")
    for img in images:
        page_text = pytesseract.image_to_string(Image.open(img))
        text += " -- " + page_text
        os.remove(img)
    return text

def extract_text_easyocr(imagepath, sep=""):
    reader = easyocr.Reader(['en'])
    text_extracted = reader.readtext(imagepath, paragraph="False")
    result = []
    for txt in text_extracted:
        result.append(txt[-1])
    return result

def get_answer(context, question, 
               cleaning_text=r'"([^"]*)"', number=1):
    time.sleep(30)
    prompt = context + " What is the" + question + " in the above text? Say just the" + question + " in quotation marks or parentheses."
    answer = openai.ChatCompletion.create(
        messages=[{
            "role": "assistant",
            "content": prompt
        }],
        temperature=0,
        max_tokens=300,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
        model="gpt-3.5-turbo"
        )["choices"][0]["message"]["content"]
    if cleaning_text == None:
        return answer
    answer = re.findall(cleaning_text, answer)
  
    if number == "all":
        number = len(answer)
    if number == 0:
        return None
    if len(answer) < number:
        return None
    else:
        if number == 1:
            return answer[0]
        else:
            return answer[: number]