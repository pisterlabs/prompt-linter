import json
import os

import openai

openai.organization = "org-mopAysnn5W9mn2Iqp8q4BNAT"
openai.api_key = os.getenv("OPENAI_API_KEY")

if __name__ == "__main__":
    with open("openai-model-list.json", "w", encoding="utf8") as file:
        json.dump(openai.Model.list(), file)
