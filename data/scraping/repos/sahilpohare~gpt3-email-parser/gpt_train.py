import openai
from gpt import Example
from gpt import GPT
from converter import convert
import os
import requests
from dataretrieve import get_content, get_answer
import json


class GPTTrain(object):
    def __init__(self, data, c_type,domain="github"):
        self.domain = domain
        self.data = data
        self.c_type = c_type

    def read_html(self, path):

        with open(path, "r") as f:
            resp = f.read()
            f.close()
            resp = convert(resp)
        return resp

    def parse(self, test):
        gpt3 = GPT(engine="davinci", temperature=0.9, max_tokens=200)

        for instance in self.data["data"]:
            gpt3.add_example(Example(get_content(instance, self.c_type), get_answer(instance)))
        output = gpt3.get_top_reply(test).replace("output: ", "").strip()

        output = output[: output.find("}") + 1]
        output = output.replace("'", '"')
        key_list = [list(x["answer"].keys()) for x in self.data["data"]]
        return output, key_list
