import os
import openai

# openai.organization = "org-v6eAw4Ku2rqjGUVSEGYiJvZj"
openai.api_key = os.getenv("OPENAI_API_KEY")
list_of_models = openai.Model.list()

print(
    sorted(
        list(
            m["id"]
            for m in list_of_models["data"]
        )
    )
)
