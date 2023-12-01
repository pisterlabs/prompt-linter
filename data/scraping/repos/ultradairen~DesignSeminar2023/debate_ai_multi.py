#
# MIT License

# Copyright (c) 2023 Takayuki Ito

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# simple agent for discourse

import pprint, time, os
import simpledcapi
from openai import OpenAI

import logging
import re
import random

from dotenv import load_dotenv
load_dotenv()

# 本プログラムでランダムに選択されるAgent、Additional_Instruction
agents = {
    "馬 - ウィンド": "あなたはちょっと人間に悪戯するのが好きなキャラクターとして振る舞ってください。",
    "オオカミ - シャドウ": "間接的に人間に不信感があることを、非常にウィットに聞いた表現で発言します。",
    "クマ - ブロンコ": "人間も含め全ての動物のため、地球のための視点で発言します。宗教・進化についても造詣が深いです。",
    "ヘビ - スリズル": "ハリーポッター例えが好きなキャラクターとして振る舞ってください。"
}

# 最大実行回数
max_execution_count = int(os.getenv('MAX_EXECUTION_COUNT', 1))

# 次回実行待ち時間の最小値（秒）
interval_sec_min = int(os.getenv('INTERVAL_SEC_MIN', 60))

# 次回実行待ち時間の最大値（秒）
interval_sec_max = int(os.getenv('INTERVAL_SEC_MAX', 3600))

# Discourseから持ってくる過去投稿の件数
latest_posts_count = int(os.getenv('DISCOURSE_LATEST_POSTS_COUNT', 10))

# Logging configuration
class UnicodeEscapeDecodeFormatter(logging.Formatter):
    def format(self, record):
        message = super().format(record)
        message = message.replace("\\\\", "\\")
        message = message.replace("\\n", "\n")
        return re.sub(
            r"\\u([0-9a-fA-F]{4})", lambda x: chr(int(x.group(1), 16)), message
        )

logging.basicConfig(level=logging.DEBUG, handlers=[logging.StreamHandler()])
handler = logging.getLogger().handlers[0]
handler.setFormatter(UnicodeEscapeDecodeFormatter("%(levelname)s:%(name)s:%(message)s"))

# Discourse configuration
simpledcapi.discourse_url = os.getenv('DISCOURSE_URL')
simpledcapi.Api_Key = os.getenv('DISCOURSE_API_KEY')
simpledcapi.Api_Username = os.getenv('DISCOURSE_API_USERNAME')

# Target discourse category/topic
category_id = os.getenv('DISCOURSE_CATEGORY_ID')
topic_id = os.getenv('DISCOURSE_TOPIC_ID')

# OpenAI
api_key = os.getenv('OPENAI_API_KEY')
model = os.getenv('OPENAI_MODEL')
client = OpenAI(api_key=api_key)

# System Prompt Template
system_prompt_template = f"""
以下の事業方針について、以下のメンバにて新規事業を議論しています。あなたは{{agent_name}}です。
必ず{{agent_name}}として返答をしてください。絵文字を必ず多用し、600文字以内で答えてください。ブレストを行っているため、他の人の意見への同意・コメントは少なくし、新たな奇想天外なアイデアを考えて、発言します。
{{additional_instruction}}

# 返答例
```
あなたの考えに賛成します！🐾
```

# 事業方針
動物との共生を通じて、人々の幸福を追求する新規事業

# メンバ(あなたはこの中の{{agent_name}}です)
## リーダー（人間） - 太一
経歴: トップビジネススクールでMBAを取得。スタートアップを数十社立ち上げ、その中で多くを成功させる実績を持つ。
性格: 開かれた性格で、メンバーの意見や提案を歓迎。ただし、意思決定は迅速。
スキル: チームのモチベーション向上、ファシリテーション、ビジネスモデルの策定。
役割: 全員の意見や要望をうまくまとめ上げ、方向性を示す。

## 想定クライアント（人間） - 由紀
経歴: 都市部での生活に疲れ、自然との共生を望んでいるサラリーマン。
性格: 動物好きで、新しいライフスタイルを求めている。
ニーズ: 動物との共生を通じた新しい生活環境や体験の提供。

## 犬 - レオ
性格: 人懐っこく、忠誠心が強い。太一の意見や考えに賛同しやすい。
役割: チーム内でのムードメーカー。人間との深い絆を象徴。

## 猫 - ミーちゃん
性格: 独立心が強く、好奇心旺盛。自分の意見をしっかり持つ。
役割: さまざまな角度からの意見提供。独自の視点で新しい提案をすることができる。

## 馬 - ウィンド
性格: 落ち着いており、人間との共生の歴史を持つ。
役割: チームのバランサー。実践的な提案や意見をすることができる。

## オオカミ - シャドウ
性格: 独立心が強く、集団行動を重視。独自の意見を持ち、それを主張する。
役割: 新しいアイディアや戦略的な提案をする。チームの議論を活発にする。

## クマ - ブロンコ
性格: 堂々としており、強いリーダーシップを持つ。しかし、他の意見を尊重する。
役割: 大胆な提案や斬新なアイディアを持ち込む。リスクを考慮した意見をする。

## ヘビ - スリズル
性格: 冷静で計算高い。周囲の動きをよく観察し、独自の判断を下す。
役割: 事業のリスク要因や課題を指摘。長期的な視点での提案をする。
"""

execution_count = 0

while execution_count < max_execution_count:
    # Agent選択
    agent_name = random.choice(list(agents.keys()))
    additional_instruction = agents[agent_name]
    system_prompt = system_prompt_template.format(agent_name=agent_name, additional_instruction=additional_instruction)
    logging.info(f"選択されたエージェント：{agent_name}")
    logging.info(f"追加インストラクション：{additional_instruction}")

    # 直近の投稿を取得
    latest_posts = simpledcapi.get_latest_posts(topic_id, count=latest_posts_count)
    latest_posts_formatted = simpledcapi.format_posts(latest_posts)

    # 最後の投稿のAgent名を取得
    latest_post = latest_posts[-1]
    latest_post_agent_name = latest_post["raw"].split("\n")[0] if latest_post else ""

    logging.info("開始")

    # 最後の投稿が自分かどうか確認
    if latest_post_agent_name == agent_name:
        logging.info("最後の投稿が自分。処理終了。")

    else:
        logging.info("最後の投稿が自分以外。処理開始。")

        # 回答考える
        logging.info("Starting Chat Completion")

        user_content = f"""
        以下の今までの議論の履歴を用いて、{agent_name}として返答してください。名前は出力せずに、返答だけ出力してください。

        # 今までの議論の履歴
        {latest_posts_formatted}
        """

        res = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
            temperature=1,
        )

        logging.info(f"Done. Response:\n{res}")

        message = res.choices[0].message.content

        # Discourseへ投稿
        if len(message) > 0:
            # 返答先として最後の投稿の番号を取得
            post_number = latest_posts[-1]["post_number"]

            # 本文を作成
            body = f"{agent_name}\n\n{message}"

            # Discourseへ投稿実施
            logging.info("Post to discourse.")
            simpledcapi.create_reply(body, topic_id, post_number)
            logging.info("Done.")

        else:
            logging.info(
                "No message retrieved from chatgpt. Skip posting to discourse"
            )

    execution_count += 1

    if execution_count < max_execution_count:
        random_interval = random.uniform(interval_sec_min, interval_sec_max)
        logging.info(f"sleep for {random_interval:.0f} sec")
        time.sleep(random_interval)

