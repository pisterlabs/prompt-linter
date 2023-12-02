import os
import openai
import json
import time

openai.api_key = "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"
api_key = os.getenv("OPENAI_API_KEY")
if api_key is None:
    print("OPENAI_API_KEY environment variable not found")
else:
    api_key = "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"


def chat(msg):
    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": msg}
        ]
    )
    return completion.choices[0].message


def load_notes(fn="notes.txt"):
    with open(fn) as f:
        return f.read()


def save_chat(output):
    fn = str(time.time()) + ".txt"
    with open(fn, "w") as f:
        f.write(output["content"])
    print("Saved chat to {}".format(fn))
    print(output)


notes = load_notes("/home/boittier/Documents/"
                   "phd/ff_energy/experiments/ff_fit/notes.txt")

output = chat("Reply with only a json file, no other formatting,"
              "the file will be json dictionary from"
              "all combinations the following: {}".format(notes))
print(output)
save_chat(output)


