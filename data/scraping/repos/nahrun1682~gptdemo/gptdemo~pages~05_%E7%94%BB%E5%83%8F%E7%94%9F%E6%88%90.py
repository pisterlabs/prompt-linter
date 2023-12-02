import streamlit as st
import openai
import requests
from PIL import Image
from io import BytesIO
import os

# Streamlitアプリのタイトル
st.title("DALL-E 画像生成アプリ")

# APIキーの設定（環境変数から取得することを推奨）
openai.api_key = os.environ["OPENAI_API_KEY"]

# DALL-Eによる画像生成関数
def generate_image_N(prompt, n=1,size="1024x1024"):
    response = openai.Image.create(
        prompt=prompt,
        n=n,  # ユーザーが選択した画像の数
        size=size  # 画像サイズ
    )
    return [data['url'] for data in response['data']]

# # ユーザー入力フィールド
user_input = st.text_input("画像を生成したいテキストを入力してください", "white cat")

# 画像のサイズを選択するためのラジオボタン
image_size = st.radio(
    "画像のサイズを選択してください",
    ('256x256', '512x512', '1024x1024'),
    index=0  # デフォルトは '1024x1024'
)

# 画像の数を選択するためのスライダー
number_of_images = st.slider("生成する画像の数", min_value=1, max_value=5, value=1)

# 画像生成ボタン
if st.button("画像生成"):
    # 画像を生成
    image_urls = generate_image_N(user_input, number_of_images, image_size)

    # 生成されたすべての画像を表示
    for image_url in image_urls:
        response = requests.get(image_url)
        img = Image.open(BytesIO(response.content))
        st.image(img)


sec_prompt="""
{
  "Art Style": "Anime/Manga",
  "Color Palette": "Vibrant, cool tones",
  "Background": "Japanese park with cherry blossoms",
  "Image Size": "768x768",
  "Character": {
    "Type": "Female Teenager",
    "Expression": "Joyful",
    "Pose": "Jumping or Running",
    "Outfit": {
      "Top": "Pink loose blouse with floral pattern",
      "Bottom": "White flowy skirt"
    },
    "Race": "Anime-style ambiguous",
    "Appearance": {
      "Eyes": "Large almond-shaped, blue/pink with hints of purple",
      "Nose": "Small",
      "Mouth": "Smiling, light pink lips",
      "Eyebrows": "Delicate, light gray",
      "Hair": "Silver/white, long, side-parted, flowing with side-swept bangs",
      "Physical": "Slender, fair skin",
      "Accessories": [
        "Ornamental hairpiece on right",
        "Cherry blossom hairpin on left"
      ],
      "Distinctive": "Galaxy/star pattern in hair, mismatched eye colors"
    }
  }
}
"""