#!/usr/bin/env python
# coding=utf-8

import os, sys, json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "lib"))
from splunklib.searchcommands import dispatch, StreamingCommand, Configuration, Option, validators
import openai


def getConfig():
    if os.path.exists(os.path.join(os.path.dirname(__file__), "..", "local", "config.json")):
        with open(os.path.join(os.path.dirname(__file__), "..", "local", "config.json")) as file:
            config = json.load(file)
            return config
    else:
        with open(os.path.join(os.path.dirname(__file__), "..", "default", "config.json")) as file:
            config = json.load(file)
            return config

config = getConfig()
openai.api_key = config["api_key"]

@Configuration()
class SplunkGPT(StreamingCommand):


    def stream(self, records):


        for record in records:
            if "prompt" in record:
                completion = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": "You are a senior IT engineer."},
                        {"role": "user", "content": record["prompt"]}
                    ]
                )

                record["response"] = completion.choices[0].message.content
                yield record
            else:
                yield record


dispatch(SplunkGPT, sys.argv, sys.stdin, sys.stdout, __name__)