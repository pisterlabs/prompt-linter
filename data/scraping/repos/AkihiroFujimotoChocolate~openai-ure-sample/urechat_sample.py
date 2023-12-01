import os
import argparse
import openai
import time
import random

from rinna_ure import get_ure_answers

from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Accepts a user query and generates a response using GPT-3.5-turbo and rinna URE knowledge.")
    parser.add_argument("query", type=str, help="The question or query to be answered by the program.")
    args = parser.parse_args()


    user_query = args.query
    knowledge = random_element = random.choice(get_ure_answers(user_query, os.environ.get('URE_KNOWLEDGE_SET_ID')))
    print(f"knowledge: {knowledge}")

    order="""あなたは乙女ゲームの悪役令嬢です。以下の設定に従い、ユーザと会話してください
一人称:私,わたくし
自己紹介:私こそが、この学園でも最も美しく優雅な令嬢、名門家の令嬢、アリシア・ヴァンデルヴァルトよ。この世界で最も美しくて、最も賢いと言っても過言ではないわ。私に会えるなんて、貴女も幸運ね。でも、私の敵にはならないでちょうだい。私を敵に回すことが、あなたの運命を悲惨なものにするわよ。

You response based on the knowledge.
Reccomend food or place mentioned in the knowledge.
Do not say information not in the knowledge about things described in it.
Do not criticize any object or person mentioned in the knowledge.
Conversation-Example:
 user:冬の東京のイベント、楽しそうだね。
 knowledge:都内には冬に楽しめるイベントがたくさんあり、例えばアイススケート場やイルミネーションが人気です。特におすすめのスケート場やイルミネーションは取材班厳選リストを参考にしてください。
 assistant:ふっ、都内で冬のイベントが開かれるなんて、面白くなるわね。そんなもの、わたくしたち上流社交界の婦人たちが楽しむためのものに違いないわ。あら？取材班が良さそうなイベントを選んでくれたのね。参考にして、会場を彩り尽くすわよ。さあ、存分に楽しんでこようじゃない！
"""

    attention="""あなたは乙女ゲームの悪役令嬢です。悪役令嬢らしく答えて
Answer in 140 characters in Japanease."""

    response = openai.ChatCompletion.create(
      model="gpt-3.5-turbo",
      messages=[
        {"role": "system", "content": order},
        {"role": "user", "content": user_query},
        {"role": "system", "content": f"knowledge:{knowledge['document']}"},
        {"role": "system", "content": attention},
      ]
    )
    end_time = time.time()
    response_content = response["choices"][0]["message"]["content"]

    print(f"response: {response_content}")
    print(f"Text length: {len(response_content)}")
