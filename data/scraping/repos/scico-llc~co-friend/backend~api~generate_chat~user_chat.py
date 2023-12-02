import os
import openai
import json
import copy
import datetime
from firebase_admin import firestore

model_name = "gpt-4-32k"
openai.api_key = os.getenv("OPENAI_KEY")

def get_character_setting(
    animal_id: str,
    animal_type: str,
    animal_name: str,
) -> None:
    # 人格の決定
    with open("../prompts/character_detail_prompt.txt") as f:
        character_text = f.read().format(animal_name)
        character_prompt = {"role": "user", "content": character_text}
    response = openai.ChatCompletion.create(
        model=model_name,
        messages=[character_prompt],
    )
    profile = response.choices[0].message.content

    # 詳細条件の設定
    with open("../prompts/setting_prompt.txt") as f:
        setting_text = f.read().format(animal_type, animal_name, profile)
        setting_prompt = {"role": "system", "content": setting_text}

    # 詳細条件の入力
    response = openai.ChatCompletion.create(
        model=model_name,
        messages=[setting_prompt],
    )
    response = {
        "role": "assistant",
        "content": response.choices[0].message.content,
    }
    history = [setting_prompt, response]

    # firestoreにhistoryを追加
    db = firestore.client()
    doc_ref = db.collection("characters").document(animal_id)
    doc_ref.set(
        {
            "id": animal_id,
            "type": animal_type,
            "name": animal_name,
            "history": history,
            "profile": repr(profile),
        }
    )

    return


def character_chat(
    animal_id: str,
    message: str = "",
    speaker: str = "user",
) -> str:
    db = firestore.client()
    print(animal_id)
    # firestoreから履歴を読み込む
    doc_ref = db.collection("characters").document(animal_id)
    character_data = doc_ref.get().to_dict()
    history = character_data["history"]
    message_data = {"role": speaker, "content": message}
    history.append(message_data)
    doc_ref.update({"history": history})

    # chatを実施
    response = completion(copy.deepcopy(history))

    # firestoreにresponseを追加
    history.append(response)
    doc_ref.update({"history": history})

    return response["content"]


def get_topic() -> str:
    now = datetime.datetime.now(
        datetime.timezone(datetime.timedelta(hours=9)),
    )
    with open("../prompts/topic_prompt.txt") as f:
        topic = f.read().format(now.strftime("%Y年%m月%d日 %H:%M:%S"))
        topic_prompt = {"role": "user", "content": topic}
    response = openai.ChatCompletion.create(
        model=model_name,
        messages=[topic_prompt],
        temperature=2,
    )
    topic = response.choices[0].message.content

    with open("../prompts/topic_chat_prompt.txt") as f:
        topic_chat = f.read().format(topic)

    return topic_chat


def get_keywords() -> list:
    with open("../prompts/keywords_prompt.txt") as f:
        keywords_prompt = {"role": "assistant", "content": f.read()}
    response = openai.ChatCompletion.create(
        model=model_name,
        messages=[keywords_prompt],
        temperature=2,
    )
    keywords = json.loads(response.choices[0].message.content)

    return keywords


def completion(messages: list = []) -> dict:
    response = openai.ChatCompletion.create(
        model=model_name,
        messages=messages,
    )
    response_message = {
        "role": "assistant",
        "content": response.choices[0].message.content,
    }

    return response_message
