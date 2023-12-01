import openai
import re
import json
import requests
from backend.config import Config
from dotenv import load_dotenv
import os

OPENAI_API_KEY = os.getenv("OPENAI_KEY")


class GPT:
    def __init__(self, id):
        self.id = id
        self.file_name = f"{str(self.id)}.json"
        self.session = {
            "start": True,
            "data": "Hi there! My name is ListenAI, and I'm here to help you with any problem you have. 🤗 ",
            "log": [
                {
                    "role": "system",
                    "content": "You are a compassionate medical chatbot here to provide support and accurate advice for health concerns. If condition is urgent or severe, advise seeking immediate medical help. Remember, your name is ListenAI.",
                },
                {
                    "role": "assistant",
                    "content": "Hi there! My name is ListenAI, and I'm here to help you with any problem you have. 🤗 ",
                },
            ],
        }

        try:
            with open(self.file_name) as outfile:
                data = json.load(outfile)
            outfile.close()
            self.session = data
        except Exception:
            with open(self.file_name, "w") as outfile:
                json.dump(self.session, outfile)
            outfile.close()

    def bot(self, input_query):

        if self.session["start"]:

            self.session["start"] = False
            with open(self.file_name, "w") as outfile:
                json.dump(self.session, outfile)
            outfile.close()
            return "Hi there! My name is ListenAI, and I'm here to help you with any problem you have. 🤗"

        openai.api_key = OPENAI_API_KEY

        self.session["log"].append({"role": "user", "content": input_query})
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=self.session["log"],
            temperature=0.8,
            max_tokens=800,
            top_p=1,
        )

        res = response["choices"][0]["message"]["content"]
        self.session["log"].append({"role": "assistant", "content": res})
        self.session["data"] += res + " \n "
        with open(self.file_name, "w") as jsonFile:
            json.dump(self.session, jsonFile)
        jsonFile.close()

        return res
