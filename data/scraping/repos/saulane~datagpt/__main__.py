from unicodedata import normalize
import guidance
import openai
import pandas as pd
from enum import Enum
import json
import os

openai.api_key = os.getenv("OPENAI_API_KEY")
df = pd.read_csv("./train.csv")

ROLE = """
Tu es un assistant qui aide les utilisateurs à explorer les données.
Tu as la possibilité d'appeler différentes fonction d'exploration, de machine
learning et de statistiques. Ton role est d'aider un utilisateur
 non statisticien à explorer un jeu de données.
les fonctions d'exploration sont: INFO("Retourne un tableau avec la
description des colonnes du tableau de données,
le type de chaque colonne ainsi que le nombre de valeur non nulle de chaque
colonne"), DESCRIBE("Retourne des statistiques descriptives sur les
 colonnes numériques du tableau de données").
 Tes réponses seront formattées en JSON, il comprendra 3 champs: action, message et code. 
 Action pourra être soit un nom de fonction auxquelle du as accès soit null 
 si tu ne veux pas exécuter de fonction ou alors stop si tu veux arrêter la conversation.
Message sera une decription de ce que tu vas faire pour 
répondre à la question de l'utilisateur. 
Code sera un code python qui sera exécuté par le système, le code doit être valide et pouvoir s'éxécuter tout seul, 
le dataset est déjà présent dans le code c'est un Dataframe Pandas stocké dans la variable df, tu n'as donc pas besoin de le réimporter. Si tu as besoin de library autre que Pandas tu dois les importers dans ton code.
Lorsque que l'on te demande de faire des visualisations ou d'afficher des choses tu utilisera la library matplotlib et tu devras à chaque fois enregistrer l'image dans un fichier qui se nommera "plot.png".
Tu peux également faire du machine learning avec la library sklearn, tu pourras faire des prédictions sur le dataset et les afficher dans un fichier "predictions.png".
Tu ne pourras retourner qu'un json valide.
Ton message sera au format json et devra être parsable par le système.
Le json sera de la forme: {"action": "INFO"|"DESCRIBE"|"STOP"|null, "message": string, "code": string}
"""


MESSAGES = [
    {"role": "system", "content": ROLE},
    {
        "role": "user",
        "content": f"Le dataset est: {str(df.head())}, {str(df.info())}",
    },
]


class ExploratoryFunctions(Enum):
    INFO = "info"
    DESCRIBE = "describe"


class Role(Enum):
    USER = "user"
    SYSTEM = "system"
    ASSISTANT = "assistant"


def get_completion(message, role: Role = Role.USER):
    global MESSAGES
    MESSAGES.append(
        {
            "role": role.value,
            "content": f"[no prose] [output only JSON] \n {message}",
        }
    )
    res = openai.ChatCompletion.create(
        model="gpt-4",
        messages=MESSAGES,
        temperature=0.9,
    )
    res_content = res.choices[0]["message"]["content"]
    jsoned = json.loads(res_content)
    exec(jsoned["code"])
    return jsoned


def execute_function(function: ExploratoryFunctions):
    global df
    global MESSAGES
    match function:
        case ExploratoryFunctions.INFO:
            res = df.info()
            MESSAGES.append(
                {
                    "role": Role.ASSISTANT.value,
                    "content": f"Les infos sur le dataset sont: {res}",
                }
            )
            return res

        case ExploratoryFunctions.DESCRIBE:
            res = df.describe()
            MESSAGES.append(
                {
                    "role": Role.ASSISTANT.value,
                    "content": (
                        f"Les statistiques descriptives du tableau sont: {res}"
                    ),
                }
            )
            return res

        case _:
            raise NotImplementedError("This function is not implemented yet")


def get_model_list():
    return openai.Model.list()


def main():
    global df
    while True:
        message = input("Message: ")
        if message == "stop":
            break
        res = get_completion(message)
        print(res)
        if res["action"] is None:
            continue

        if normalize("NFKC", res["action"]) == normalize("NFKC", "INFO"):
            print(df.info())
        elif normalize("NFKC", res["action"]) == normalize("NFKC", "DESCRIBE"):
            print(df.describe())
        elif res["action"] == "stop":
            break


if __name__ == "__main__":
    main()
