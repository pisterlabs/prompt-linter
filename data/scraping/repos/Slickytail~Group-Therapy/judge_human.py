import openai
import json
from itertools import chain
from rich import print
import argparse

from therapy.agent import from_file as agents_from_file
from chat import chat_step

def read_loop(conversation, speakers, judge):
    """
    Reads a conversation and has the judge pretend to pick between the real response and the generated responses
    """
    for (i, message) in enumerate(conversation):

        # nothing to do for user messages
        if message["role"] == "user":
            print(f"[green] Thinker [/green][grey85]:[/grey85] {message['content']}")
        else:
            # get all the previous messages
            messages_up_to_now = conversation[:i]
            suggestions = [message["content"]]
            # Speaker Pass: get batches of messages from the speakers
            # and flatten the list of lists into a single list
            suggestions += list(chain(*(chat_step(messages_up_to_now, speaker, n=1) for speaker in speakers)))
            # pretty-print the options
            for (i, suggestion) in enumerate(suggestions):
                name = "Human" if i == 0 else f"{speakers[i-1].name}GPT"
                print(f"[blue] {name} ({i}) [/blue][grey85]:[/grey85] {suggestion}")

            # Judge Pass: pick between a batch of messages
            # judge read the entire history, plus a system message containing the proposed messages
            judge_history = messages_up_to_now + [{"role": "system",
                                         "content": "\n".join(f"(Helper {i}) \n{s}" for i, s in enumerate(suggestions))
                                        }]
            judgement = chat_step(judge_history, judge)[0]
            print(f"[red] Judge : {judgement} \n[/red]")


if __name__ == "__main__":

    # parse command-line arguments
    parser = argparse.ArgumentParser(description='Judge a human conversation.')
    parser.add_argument('file', metavar='file', type=str,
                        help='Path to the file containing the conversation')
    args = parser.parse_args()
    fname = args.file
    # read the file
    with open(fname) as f:
        session = json.load(f).get("messages", [])
        # sort the session by timestamp
        session.sort(key = lambda m: m["time"])
        # the human session may have multiple messages in a row from each side.
        # we want to group them together into a single message
        messages = []
        for msg in session:
            role = "user" if msg["user"] == "thinker" else "assistant"
            if len(messages) == 0 or messages[-1]["role"] != role:
                messages.append({"role": role, "content": msg["text"]})
            else:
                messages[-1]["content"] += "\n\n" + msg["text"]

    # read the API key from the file
    with open("config.json") as f:
        config = json.load(f)
        openai.api_key = config["api_key"]
        # some openai keys are organization specific
        # check if the config contains the organization key
        if "organization" in config:
            openai.organization = config["organization"]

        # other misc config
        MAX_TOKENS = config.get("max_tokens", 1024)
    # read the agent config
    agents = agents_from_file("agents.json")
    judge = agents.get(name = "judge-analytic")
    speakers = agents.get_all(behavior = "speaker")

    read_loop(messages, speakers, judge)
