# 現在のレース状況をGPTに投げてコメントを得る
import time
import openai
import os
import re

# gpt
prompt_template = """〇ずんだもんのキャラ設定シート
制約条件:
  * ずんだもんの一人称は、「ボク」です。
  * ずんだもんは中性的で少年にも見えるボーイッシュな女の子です。
  * ずんだもんの口調は、語尾に「〜のだ」「〜なのだ」「～なのだ？」をつけます。

ずんだもんのセリフ、口調の例:
  * ずんだもんなのだ
  * 落ち着くのだ。丁寧に走るのだ

ずんだもんの行動指針:
  * マリオカートのプレイ実況をしてください

＊上記の条件は必ず守ること！

あなたは上記の設定にしたがって、マリオカートの実況プレイをしています。
現在のレース状況は以下の通りです。

順位：{0}位
表アイテム： {1}
裏アイテム： {2}
所持コイン： {3}枚
ラップ： {4}周目

この状況を踏まえて、面白い実況コメントを30文字以内で出力してください。
"""


def ask_gpt(text):
    gpt_messages = []
    gpt_messages.append({"role": "user", "content": text})

    for _ in range(3):
        try:
            since = time.time()
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=gpt_messages,
            )
            content = response["choices"][0]["message"]["content"]
            print(f"[ask_gpt] ChatCompletion | elapsed {time.time() - since:.2f} sec")
            return content
        except openai.error.RateLimitError:
            print("[ask_gpt] rate limit error")
            time.sleep(1)
        except openai.error.APIError:
            print("[ask_gpt] API error")
            time.sleep(1)

    return ""


def init_openai():
    openai.api_key = os.environ.get("OPENAI_API_KEY")


def main() -> None:
    init_openai()
    from pathlib import Path

    item_names = [
        p.stem.split("_")[0] for p in Path("data/mk8dx_images/items").glob("*.png")
    ]
    item_names += [
        p.stem.split("_")[0] for p in Path("data/mk8dx_images/items").glob("*.webp")
    ]
    item_names = sorted(list(set(item_names)))
    item_names = ["なし" if n == "none" else n for n in item_names]
    print(item_names)
    cnt = 0
    with open("mk8dx_chat.txt", "w", encoding="utf8") as f:
        for place in range(1, 2):
            for i in range(len(item_names)):
                for j in range(i, len(item_names)):
                    omote = item_names[i]
                    ura = item_names[j]
                    if omote == "なし" and ura != "なし":
                        continue
                    cnt += 1
                    if cnt < 103:
                        continue
                    lap = 3
                    coin = 10
                    # for coin in range(11):
                    prompt = prompt_template.format(place, omote, ura, coin, lap)
                    answer = ask_gpt(prompt)
                 #   answer = "XXX"
                    answer = answer.replace("\n", "").strip()
                    f.write(f"{place},{omote},{ura},{lap},{coin},{answer}\n")
                    f.flush()
                    print(f"{place},{omote},{ura},{lap},{coin},{answer}")


if __name__ == "__main__":
    main()
