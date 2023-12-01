import openai
import os


from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())

api_key = os.getenv('OPENAI_API_KEY')

import base64
import requests

# OpenAI API Key
#api_key = "YOUR_OPENAI_API_KEY"

# Function to encode the image
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

# Path to your image
image_path = "bf1.png"

# Getting the base64 string
base64_image = encode_image(image_path)

headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {api_key}"
}

payload = {
    "model": "gpt-4-vision-preview",
    "messages": [
      { 
        "role": "system",
        "content": 
          {
            "type": "text",
            "text": '''你是一个计算机专家，图片是位操作的问题,包括移位和循环移位，请识别图像中的递归表达式，然后回答图片上的问题。
            Bit-String Flicking规则：
Left Circular Shift (LCIRC)
（LCIRC-x abcdef）：LCIRC表示左循环移位。x表示移位的位数，abcdef表示二进制的位。LCIRC-x是操作码，abcdef就是左移操作的二进制的操作数，每个字母表示一个bit位。
操作方法是：将二进制数的左边x位（从左到右）切出来，变成2组，比如x是4，就把abcd这4位切出来，变成了两组：abcd和ef。然后把abcd组拼接到ef组后面即可，变成efabcd即可。x是其他值以此类推。

Right Circular Shift (RCIRC)
（RCIRC-x abcdef）：RCIRC表示右循环移位。x表示移位的位数，abcdef表示二进制的位。RCIRC-x是操作码，abcdef就是左移操作的二进制的操作数，每个字母表示一个bit位。
操作方法是：将二进制数的右边x位（二进制的最低x位）切出来，比如x是3，就把def三位切出来，变成两组 abc和def，然后把切除的def组放到abc组的前面（左边）就是defabc。x是其他值以此类推。

Order of Precedence：
    The order of precedence (from highest to lowest) is: NOT; SHIFT and CIRC; AND; XOR; and finally, OR. In other words, all unary operators are performed on a single operator first.
'''
          },
        "role": "user",
        "content": [
          {
            "type": "text",
            "text": "这是一个二进制的位操作的问题,请识别图片上位操作表达式，如果图片上有问题，请回答图片上的问题"
          },
          {
            "type": "image_url",
            "image_url": {
              "url": f"data:image/jpeg;base64,{base64_image}"
            }
          }
        ]
      }
    ],
    "max_tokens": 600
}

response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
response_json=response.json()

content = response_json['choices'][0]['message']['content']
print(content)