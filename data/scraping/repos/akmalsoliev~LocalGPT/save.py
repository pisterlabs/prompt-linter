import json

from langchain.schema import messages_to_dict


def save_messages(messages:list):
    list_msg = messages_to_dict(messages)
    with open("config/settings.json", "r") as file:
        settings_json = json.load(file)

    file_name = None
    if len(list_msg) > 2:
        for dict_msg in list_msg:
            if dict_msg["type"] == "human":
                cleaned_name = (
                    dict_msg["data"]["content"][:30].capitalize().split()
                )
                file_name = " ".join(cleaned_name)
                break
    
    CHAT_PATH = settings_json.get("CHAT_PATH", "io/chat")
    if file_name:
        file_dir = f"{CHAT_PATH}/{file_name}.json"
        with open(file_dir, "w") as json_file:
            json.dump(list_msg, json_file)
