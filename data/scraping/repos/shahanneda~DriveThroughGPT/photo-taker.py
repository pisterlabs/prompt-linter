import cv2
import time
import os
from openai import OpenAI
from dotenv import load_dotenv
import base64


load_dotenv()  


folder = "images"
image_path = f"{folder}/image.png"

cam_port = 0
delay = 5

def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')

def ask_gpt_img():
  base64_image = encode_image(image_path)
  client = OpenAI()
  response = client.chat.completions.create(
    model="gpt-4-vision-preview",
    messages=[
      {
        "role": "user",
        "content": [
          {"type": "text", "text": "What’s in this image?"},
          {
            "type": "image_url",
            "image_url": {
              "url": f"data:image/jpeg;base64,{base64_image}"
            },
          },
        ],
      }
    ],
    max_tokens=300,
  )
  print(response.choices[0])
  
cam = cv2.VideoCapture(cam_port) 
i = 0
fps = 24
frame_count = 0
save_interval = 1
while cam.isOpened():
    ret, frame = cam.read()
    if ret:
        frame_count += 1

        if frame_count % (fps * save_interval) == 0:
            print("saving frame!!")
            cv2.imwrite(image_path, frame)
            frame_count = 0
    else:
        break
  
# while True:
# 		print("taking photo!")
# 		result, image = cam.read() 
# 		if not result:
# 				print("No image detected. Please! try again") 
# 				time.sleep(delay)
# 				continue

# 		print("writing photo")
# 		# os.remove(image_path)
# 		cv2.imwrite(f"images/{i}.png", image)
# 		i += 1
# 		# print("asking gpt")
# 		# ask_gpt_img()

# 		time.sleep(delay)
