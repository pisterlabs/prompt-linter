import sys
from dotenv import load_dotenv
import os
from langchain.chat_models import ChatOpenAI
from langchain import PromptTemplate,LLMChain
from langchain.prompts.chat import (
    ChatPromptTemplate,
    PromptTemplate,
    SystemMessagePromptTemplate,
    AIMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage
)
import json
import requests


# 環境変数をロード
load_dotenv('.env')
OPEN_AI_API_KEY = os.getenv('OPEN_AI_API_KEY')
NOTION_TOKEN = os.getenv('NOTION_TOKEN')
print(sys.argv)

# コマンドライン引数を取得
img_to_text = sys.argv[1]
print(img_to_text)

# # エスケープされた改行を復元
# img_to_text = img_to_text.replace('\\n', '\n')


chat = ChatOpenAI(openai_api_key=OPEN_AI_API_KEY,model_name = "gpt-3.5-turbo-0613")

#システムメッセージの作成
system_template1 = "あなたは優れたマーケターです。分析結果をフォーマットの通りにJSON形式のみを出力してください。"
system_message_prompt = SystemMessagePromptTemplate.from_template(system_template1)
#ソースの入力
explanation = """
1.誤字脱字を修正し（OCRの文章なので）
2.フォーマットに沿った出力をしてください
##注意点
文章は全文出力し省略は行わないこと
フォーマットに従うこと
Json形式以外の出力はしないこと"""

how_to_output = """{
  "この投稿のタイトル": "〜",
  "この投稿の整理":{
    "見出し1": "〜",
    "内容全文1": "〜",
    "見出し2": "〜",
    "内容全文2": "〜",
    "見出し3": "〜",
    "内容全文3": "〜",
   ・・・
    "見出しN":"〜”,
   "内容全文N":"〜”,
  }
}
"""




#text1 = img_to_text
text1 = img_to_text

#ヒューマンメッセージの作成
human_template1 = "*****説明：{explanation}**********フォーマット{how_to_output}**********本文{text1}"
human_message_prompt1 = HumanMessagePromptTemplate.from_template(human_template1)



#プロンプトの完成
chat_prompt1 = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt1])
chain = LLMChain(llm=chat, prompt=chat_prompt1)

#チャットの実行
result = chain.run(explanation=explanation,how_to_output = how_to_output,text1=text1)
#print(result)

#改行を削除
result = result.replace('\n', '')
result_dict = json.loads(result)

# Your Notion token
token = NOTION_TOKEN

#APIリクエスト用のURL
url = "https://api.notion.com/v1/pages"

# Process each item in the message
def create_page_content(result_dict):
    children = []
    for index,(key, value) in enumerate(result_dict.items()):
        # Add a new header block
        data = {
            "object": "block",
            "type": "heading_2",
            "heading_2": {
                "rich_text": [{"type": "text", "text": {"content": key}}]
            }
        }
        children.append(data)
        if index == 0:
            title = value

        if isinstance(value, dict):
            for index, (subkey, subvalue) in enumerate(value.items()):
                if index % 2 == 0:
                    data = {
                        "object": "block",
                        "type": "heading_3",
                        "heading_3": {
                            "rich_text": [{"type": "text", "text": {"content": subvalue}}]
                        }
                    }
                else:
                    data = {
                        "object": "block",
                        "type": "paragraph",
                        "paragraph": {
                            "rich_text": [{"type": "text", "text": {"content": subvalue}}]
                        }
                    }
                children.append(data)
        else:
            data = {
                "object": "block",
                "type": "paragraph",
                "paragraph": {
                    "rich_text": [{"type": "text", "text": {"content": value}}]
                }
            }
            children.append(data)
    return children,title


# Headers
headers = {
    "Authorization": f"Bearer {token}",
    "Notion-Version": "2022-06-28", 
}
children_content,title = create_page_content(result_dict)
#(children_content,title)

json_data = {
    # アイコンやカバーも設定可能
    "icon": {
        "type": "emoji",
        "emoji": "🐾"
    },
    "parent": {
        "type": "database_id",
        "database_id": "6086bcc463c54a6aaddb1f3225c07117"
    },
    # プロパティ
    "properties": {
        # タイトル
        "title": {
            "title": [
                {
                    "text": {
                        "content": title
                    }
                }
            ]
        },
    },
    # 本文
    "children": children_content,
}

response = requests.post(url, json=json_data, headers=headers)
#(response.text)
