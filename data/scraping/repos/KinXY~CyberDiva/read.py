import json
import openai
import os
from openai.embeddings_utils import get_embeddings, get_embedding

def get_chara_setting_keys(name: str):
    charaInit = json.load(open(f"characters/{name}.json", "rb"))
    keys = []
    for key in charaInit.keys():
        if type(charaInit[key]) is list:
            keys.append(key)
    return keys

def get_chara_config(api_key: str):
    # get character setting
    name = json.load(open("chara.json", "rb"))["name"]
    is_embedded = json.load(open(f"characters/{name}.json", "rb"))["is_embedded"]

    if not is_embedded:
        embed_chara(name, api_key)

    charaSet = json.load(open(f"characters/{name}_embedded.json", "rb"))

    # get live2d setting
    live2d_name = json.load(open("chara.json", "rb"))["live2d"]
    embed_live2d_motions(live2d_name, api_key)
    with open(f"live2d/{live2d_name}/motions_embedded.json", "rb") as f:
        live2d_motions = json.load(f)
    charaSet["motions"] = live2d_motions
    return charaSet


# make the sayings of the character into embeddings
def embed_chara(name: str, api_key: str):
    charaInit = json.load(open(f"characters/{name}.json", "rb"))
    openai.api_key = api_key

    # get all the keys where the value is a list but not a string
    keys = get_chara_setting_keys(name)

    embedded_values = []
    # get the embeddings for the values of the keys
    for key in keys:
        values = charaInit[key]
        embeddings = get_embeddings(list_of_text=values, engine="text-embedding-ada-002")
        done_values = {"key": key, "values": []}
        # make the json of the values with the embeddings
        for i in range(len(values)):
            done_values["values"].append({"content": values[i], "embedding": embeddings[i]})
        embedded_values.append(done_values)

    # change the "is_embedded" to True
    charaInit["is_embedded"] = True
    with open(f"characters/{name}.json", "w", encoding="UTF-8") as f:
        json.dump(charaInit, f, ensure_ascii=False, indent=4)
    # output the json file with embeddings
    for embedded_value in embedded_values:
        key = embedded_value["key"]
        values = embedded_value["values"]
        charaInit[key] = values
    with open(f"characters/{name}_embedded.json", "w", encoding="UTF-8") as f:
        json.dump(charaInit, f, ensure_ascii=False, indent=4)


def embed_live2d_motions(live2d_name: str, api_key: str):
    # if the motions_embedded.json file exists, then return
    if os.path.exists(f"live2d/{live2d_name}/motions_embedded.json"):
        return

    # get the motions
    live2d_model = json.load(open(f"live2d/{live2d_name}/model.json", "rb"))
    live2d_motions = live2d_model["motions"]
    # get all the keys of the motions and transform it into a list
    live2d_motions = list(live2d_motions.keys())

    # embed the motions
    openai.api_key = api_key
    motion_embeddings = get_embeddings(
        list_of_text=live2d_motions, engine="text-embedding-ada-002"
    )
    # make the json file of the motions with the embeddings
    motions_embedded = []
    for i in range(len(live2d_motions)): 
        embedding = motion_embeddings[i]
        motions_embedded.append(
            {"content": live2d_motions[i], "embedding": embedding}
        )
    # output the json file with embeddings
    with open(f"live2d/{live2d_name}/motions_embedded.json", "w", encoding="UTF-8") as f:
        json.dump(motions_embedded, f, ensure_ascii=False, indent=4)

def get_user_config(id: str, chara_name: str):
    userInit = json.load(open(id, "rb"))
    setting = userInit["setting"]
    setting = setting.replace("CHARACTER", chara_name)
    userInit["setting"] = setting
    return userInit
