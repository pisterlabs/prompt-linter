from openai import OpenAI
from dotenv import load_dotenv
import os
load_dotenv()

client = OpenAI(
    api_key=os.getenv("OPEN_AI_KEY"),
)

response = client.images.generate(
  model="dall-e-3",
  prompt="アニメスタイルの女の子のキャラクター、大きな目、白髪、ヘッドフォン、表情は少し不機嫌そう、フルフェイスのクローズアップ",
  size= "1024x1024",
  quality="standard",
  n=1,
)

print(response.model_dump_json(indent=2))