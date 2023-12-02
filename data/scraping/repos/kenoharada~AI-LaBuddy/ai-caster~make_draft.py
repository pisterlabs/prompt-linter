import re
import json
import openai
import os
from dotenv import load_dotenv
load_dotenv()
openai.api_key = os.getenv('OPENAI_API_KEY')

# MODEL_NAME = 'gpt-3.5-turbo-16k'
MODEL_NAME = 'gpt-4'

def parse_voice(text):
    # パターンを定義します：名前と発言
    pattern = re.compile(r'(\w+): (.+?)(?=\w+:|$)', re.DOTALL)

    # 文字列から一致する部分をすべて見つける
    matches = pattern.findall(text)

    # マッチした結果を指定の形式に変換
    dialogue_list = [{'person': m[0], 'text': m[1].strip()} for m in matches]
    return dialogue_list


data = json.load(open('techcrunch.json'))

for idx, news in enumerate(data):
    system_message = f"""
    ###
    タイトル: {news['title']}
    本文: {news['article']}
    ### 
    アナウンサーは###で囲まれたニュース内容について簡潔に内容を報道します。
    アイドルは、ニュース内容に関連する事柄について1つ質問をします。
    アナウンサーはアイドルからの質問に簡潔に答え、一通り疑問が解消されたら「次の話題に移ります」と言って会話が終了します。
    一連の会話の流れを文章に起こしてください。
    フォーマットは
    アナウンサー: 発言
    アイドル: 発言
    のようにお願いします。発言は日本語でお願いします。
    """

    response = openai.ChatCompletion.create(
                    model=MODEL_NAME,
                    messages=[
                        {"role": "user", "content": system_message}
                    ],
                    temperature=0.7,
                )['choices'][0]['message']['content']
    print(response)
    print(parse_voice(response))
    dialogue_list = parse_voice(response)
    with open(f'news_dialogue_{idx}.json', 'w', encoding='utf-8') as file:
        json.dump(dialogue_list, file, ensure_ascii=False)
    break