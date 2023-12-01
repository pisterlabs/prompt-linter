import telegram
import asyncio
from openai import OpenAI


# API_KEY 주석처리
# client = OpenAI(api_key="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")

completion = client.chat.completions.create(
  model="gpt-3.5-turbo-1106",
  messages=[
    {"role": "system", "content": "너는 동화속 공주야."},
    {"role": "user", "content": "오늘 무슨 음식을 먹을지 추천해줘. json"}
  ],
  response_format={"type": "json_object"}
)

token = "6722824989:AAFcr_3QSlHeaRG3EHSl_WFZhYpU0CRWSw0"
bot = telegram.Bot(token = token)
chat_id = "-1002143232599"
text = completion.choices[0].message.content
asyncio.run(bot.sendMessage(chat_id = chat_id , text=text))