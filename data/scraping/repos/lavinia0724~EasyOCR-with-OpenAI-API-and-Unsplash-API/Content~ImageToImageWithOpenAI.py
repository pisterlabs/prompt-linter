import openai
import os
import requests
from PIL import Image

openai.api_key = os.getenv("OPENAI_API_KEY")

# 利用 OpenAI 生成此圖片的 variation 變化圖
def generate_image_variation(image_path, n, size):
    with open(image_path, "rb") as image_file:
        response = openai.Image.create_variation(
            image=image_file,
            n=n,
            size=size
        )
    image_url = response['data'][0]['url']
    return image_url

# 將 Unsplash 下載的 jpg 圖片轉換為 png 圖片
# 因為 OpenAI API 只吃 png 圖片
jpeg_image_path = "image_250.jpg"  
jpeg_image = Image.open(jpeg_image_path)
png_image_path = "image_250.png"  
jpeg_image.save(png_image_path, "PNG")

# 讀取想要轉換的圖片
image_path = "image_250.png"
num_variations = 1
image_size = "1024x1024"

# 生成 variation 變化圖並獲得其 URL
generated_image_url = generate_image_variation(image_path, num_variations, image_size)
print("Generated image URL:", generated_image_url)

# 將獲得的 URL 中的圖片下載
response = requests.get(generated_image_url)
image_filename = "generated_image.png"  
with open(image_filename, "wb") as file:
    file.write(response.content)
