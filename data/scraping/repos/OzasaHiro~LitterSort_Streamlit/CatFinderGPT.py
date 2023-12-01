# main_app.py
import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import pyheif
from pydrive2.auth import GoogleAuth
from pydrive2.drive import GoogleDrive
from datetime import datetime
from rembg import remove
from openai import OpenAI
import os
import base64
import streamlit as st
from io import BytesIO
import requests

os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
client = OpenAI()

def open_image(file):
    """open image file"""
    try:
        # for HEIC type
        if file.name.lower().endswith(".heic"):
            heif_file = pyheif.read(file.getvalue())
            image = Image.frombytes(
                heif_file.mode, 
                heif_file.size, 
                heif_file.data,
                "raw",
                heif_file.mode,
                heif_file.stride,
            )
        else:
            image = Image.open(file)
    except Exception as e:
        st.error(f"Loading Error: {e}")
        return None

    return image


def encode_image(image):
    if isinstance(image, Image.Image):
        buffered = BytesIO()
        image.save(buffered, format="JPEG")  # or "PNG", depending on your image format
        img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
        return img_str
    else:
        raise TypeError("The function requires a PIL.Image.Image object")


def main():
    st.title("猫を褒めるためのAI")
    uploaded_file = st.file_uploader("猫の写真を褒めます。写っていなくても褒めます。", type=['jpg', 'png', 'jpeg', 'heic'])

    if uploaded_file is not None:
        image = open_image(uploaded_file)
        if image:
            image = image.resize((image.width // 2, image.height // 2))
            image_bg = image
            #image = remove(image)
            img_rgb = image.convert('RGB')

            if image.mode == 'RGBA':
                r, g, b, a = image.split()
                bg_white = Image.new('RGB', image.size, (255, 255, 255))
                bg_white.paste(img_rgb, mask=a)
                image = bg_white
                
            st.image(image, caption='Uploaded_photo', use_column_width=True)
            st.write("")

            # Getting the base64 string
            base64_image = encode_image(image)
        
            # OpenAIのAPIを呼び出して、画像の説明を生成
            response = client.chat.completions.create(
                model="gpt-4-vision-preview",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "画像の中に猫を見つけたら、猫のあらゆるところを全力で褒めまくってください。猫のことはねこちゃんと呼び、尊敬を持って接してください。画像の中に猫が見つからなかったら、画像の中の猫っぽい箇所を探し出してなんとしてでも褒めてください。レスポンスのTokenは250程度を目安としてください。"},
                            {
                                "type": "image_url",
                                "image_url": f"data:image/jpeg;base64,{base64_image}",
                            },
                        ],
                    }
                ],
                max_tokens=300,
            )

            # 結果を表示
            st.write(response.choices[0].message.content)    
        
   



if __name__ == "__main__":
    main()
