import os
import glob

from langchain.chat_models import ChatOpenAI
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage
)
from dotenv import load_dotenv
load_dotenv()
chat = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")
chat.openai_api_key = os.environ['OPENAI_API_KEY']

directory_path = "."  # 取得したいディレクトリのパスを指定

# ディレクトリ以下の全ての"notes.txt"ファイルのパスを取得
notes_files = sorted(glob.glob(os.path.join(directory_path, "**/notes.txt")))
overview = ''
overview_jp = ''

# 取得したパスを表示
for file_path in notes_files:
    title = file_path.split('/')[-2]
    print(title)
    with open(file_path, "r") as f:
        text = f.read()
        overview += f'## {title}\n'
        overview += text
        overview += '\n'

        overview_jp += f'## {title}\n'
        messages = [
            SystemMessage(content="You are a helpful assistant. Please translate texts into Japanese. 英語を日本語に翻訳して下さい。"),
            HumanMessage(content=text),
        ]
        response = chat(messages)
        overview_jp += response.content
        overview_jp += '\n'

with open('overview.md', 'w') as f:
    f.write(overview)

with open('overview_jp.md', 'w') as f:
    f.write(overview_jp)