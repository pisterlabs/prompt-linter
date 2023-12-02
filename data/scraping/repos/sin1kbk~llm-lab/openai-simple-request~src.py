import os
import sys
import openai

# APIキーの設定
openai.api_key = os.environ["OPENAI_API_KEY"]

# 引数チェック
if len(sys.argv) != 2:
    print("Usage: python src.py <question strings>")
    sys.exit(1)

response = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "user", "content": sys.argv[1]},
    ],
)

print(response.choices[0]["message"]["content"].strip())
