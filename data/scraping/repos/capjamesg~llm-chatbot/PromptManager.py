import json
import os
from copy import deepcopy

import numpy as np
import openai

import config

openai.api_key = config.OPENAI_KEY

# if prompts.json not present, raise error
if not os.path.exists(f"prompts.json"):
    raise Exception(
        "prompts.json not found. You must generate a prompt with `python3 generateprompt.py` before running the web app."
    )

with open("prompts.json", "r") as f:
    prompts = json.load(f)

prompt_list = prompts["prompts"]


class Prompt:
    def __init__(self, prompt_id=prompts["latest_id"]):
        self.prompt_id = prompt_id
        self.substitutions = prompt_list[self.prompt_id]["substitutions"]
        self.date_created = prompt_list[self.prompt_id]["date"]
        self.index_id = prompt_list[self.prompt_id]["index_id"]
        self.index_name = prompt_list[self.prompt_id]["index_name"]

    def __repr__(self):
        print(f"Prompt ID: {self.prompt_id}")
        print(self.prompt)

    def seek_substitutions(self):
        print(self.substitutions)

    def raw_prompt(self):
        return prompt_list[self.prompt_id]["prompt"]

    def execute(self, substitutions={}, prompt_text="", temperature=None):
        new_prompt = deepcopy(prompt_list[self.prompt_id])

        if prompt_text != "":
            new_prompt["prompt"][-1]["content"] = prompt_text

        for message in new_prompt["prompt"]:
            for key in substitutions:
                if key in message["content"]:
                    message["content"] = message["content"].replace(
                        f"[[[{key}]]]", substitutions[key]
                    )

        if temperature is not None:
            return openai.Completion.create(
                model="gpt-3.5-turbo",
                messages=new_prompt["prompt"],
            )["choices"][0]["message"]["content"]

        print(new_prompt["prompt"])

        return openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=new_prompt["prompt"],
        )["choices"][0]["message"]["content"]

    def get_facts_and_knn(self, query, vector_index, schema, facts):
        embedded_query = openai.Embedding.create(
            input=query, model="text-embedding-ada-002"
        )

        D, I = vector_index.search(
            np.array([embedded_query["data"][0]["embedding"]]).reshape(1, 1536), 25
        )

        knn = []

        for i in I[0]:
            knn.append(schema[i]["text"])

        content_sources = [
            schema[i].get("url", None) for i in I[0] if schema[i].get("url")
        ]

        if len(content_sources) > 0:
            titles = [schema[i].get("title", schema[i]["url"]) for i in I[0]]
            dates = [schema[i].get("date", "") for i in I[0]]

            # create text that looks like this
            # [fact] (Source: [source])

            facts_and_sources = []

            for fact in facts:
                facts_and_sources.append(fact + " (Source: " + config.FACT_SOURCE + ")")

            skipped = []

            for i in range(len(knn)):
                facts_and_sources.append(
                    knn[i]
                    + ' (Source: <a href="'
                    + content_sources[i]
                    + '">'
                    + titles[i]
                    + "</a>, "
                    + dates[i]
                    + ")"
                )

            facts_and_sources_text = "\n\n".join(facts_and_sources)
            # cut off at 2000 words
            facts_and_sources_text = " ".join(facts_and_sources_text.split(" ")[:300])

            references = [
                {"url": content_sources[i], "title": titles[i]}
                for i in range(len(knn))
                if i not in skipped
            ]
        else:
            facts_and_sources_text = "\n\n".join([schema[i]["text"] for i in I[0]])
            skipped = []

            references = []

        return facts_and_sources_text, knn, references
