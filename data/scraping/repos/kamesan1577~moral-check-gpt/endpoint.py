import openai
from jinaai import JinaAI


# ModerationAPIの呼び出し
def get_moderation(msg, img_description=None):
    with open("key.secret", "r") as f:
        openai.api_key = f.read().strip()
    openai.api_base = "https://api.openai.iniad.org/api/v1"

    if img_description:
        msg = f"{msg}\n ###\nDescription of the image attached to the Tweet: {img_description}"
    else:
        msg = msg
    response = openai.Moderation.create(input=msg)
    return (msg, response["results"][0])


# 英語に翻訳してから呼び出すバージョン
def get_moderation_after_translate(msg, img_description=None, model="gpt-3.5-turbo"):
    # 普通にChatGPTに翻訳させる
    msg_translated = openai.ChatCompletion.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": f"Translate this message to English(return only translation result): {msg}",
            },
        ],
    )["choices"][0]["message"]["content"]
    return get_moderation(msg_translated, img_description=img_description)


def explain_image(img_path):
    with open("scenex.secret", "r") as f:
        jinaai_api_key = f.read().strip()
    jinaai = JinaAI(secrets={"scenex-secret": jinaai_api_key})
    descriptions = jinaai.describe(img_path)
    return descriptions["results"][0]["output"]
