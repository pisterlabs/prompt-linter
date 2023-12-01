import os

from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
from langchain.schema.messages import HumanMessage, SystemMessage

load_dotenv()


chat = ChatOpenAI(model="gpt-4-vision-preview", max_tokens=512)
result = chat.invoke(
    [
        HumanMessage(
            content=[
                {"type": "text", "text": "图片里面是什么？请详细介绍一下。"},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "https://i.postimg.cc/sxnBYb28/1.jpg",
                        "detail": "auto",
                    },
                },
            ]
        )
    ]
)


print(result.content)
