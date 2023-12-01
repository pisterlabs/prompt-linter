import os
import openai
import requests
from PIL import Image

openai.api_key = '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'

output_folder = "output"
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

pre_image = Image.open("user_input.png")
pre_image = pre_image.resize((512, 512))
pre_image.save("image.png")

pre_mask = Image.open("user_mask.png")
pre_mask = pre_mask.resize((512, 512))
pre_mask.save("mask.png")



# 圖片生成設定
response = openai.Image.create_edit(
    image = open("image.png", "rb"),
    mask = open("mask.png", "rb"),
    prompt = "A cat riding a motorcycle",
    n=5,
    size="512x512"
)
response_urls = response['data']

for i in range(len(response_urls)):
  # 將圖片的網址取出
  image_url = response_urls[i]['url']
  
  # 從url取得圖片
  response = requests.get(image_url, stream=True)
  
  # 利用PIL讀取圖片
  k = Image.open(response.raw)
  
  # 製作圖片名稱
  output_filename = os.path.join(output_folder, f"user_output_{i}.png")
  
  # 儲存圖片
  k.save(output_filename)