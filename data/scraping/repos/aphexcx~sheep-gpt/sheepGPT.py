import argparse
import difflib
import time
from typing import List, Optional

import openai
import psutil
import requests
from mlc_chat import ChatModule, ChatConfig, ConvConfig
from mlc_chat.callback import DeltaCallback

from zeroconf_listener import listener


def parse_args():
    parser = argparse.ArgumentParser(
        description="Choose the model to use for response generation."
    )
    parser.add_argument(
        "--model",
        type=str,
        default="llama2",
        choices=["llama2", "gpt-4"],
        help="The model to use for response generation.",
    )
    return parser.parse_args()


args = parse_args()

CHATMODEL = "Llama-2-70b-chat-hf-q4f16_1"
with open("system_prompt_sjobs.txt", "r") as file:
    system_prompt = file.read()

if args.model == "llama2":
    print(f"Loading {CHATMODEL}...")
    # Create a ChatModule instance
    conv_config = ConvConfig(system=system_prompt)
    config = ChatConfig(temperature=0.75, conv_config=conv_config)
    cm = ChatModule(model=CHATMODEL, chat_config=config)
    print(f"{CHATMODEL} loaded")
    print(f"Current RAM usage: {psutil.Process().memory_info().rss / 1024 ** 2} MB")
else:
    print("Using OpenAI GPT-4 for response generation.")
    openai.api_key = "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"


# Define the maximum number of retries for failed operations
max_retries = 2

local_messages = []
last_posted_thought = None


def get_messages() -> Optional[List[str]]:
    global local_messages
    print("Getting messages...")
    # Define the GET endpoint
    get_endpoint = f"http://{listener.server_ip}:8080/messages"
    for _ in range(max_retries):
        try:
            response = requests.get(get_endpoint)
            messages = [msg["str"] for msg in response.json() if msg["type"] == "D"]
            print(f"Got {len(messages)} messages")
            diff = difflib.ndiff(local_messages, messages)
            new_messages = [l[2:] for l in diff if l.startswith("+ ")]
            local_messages = messages
            return new_messages
        except Exception as e:
            print(f"Error getting messages: {e}")
    return None


class ResponseCallback(DeltaCallback):
    """Stream the output of the chat module to stdout."""

    def __init__(self, callback_interval: int = 2):
        r"""Initialize the callback class with callback interval.

        Parameters
        ----------
        callback_interval : int
            The refresh rate of the streaming process.
        """
        super().__init__()
        self.callback_interval = callback_interval
        notify_generating_thought(True)

    def delta_callback(self, delta_message: str):
        r"""Stream the delta message directly to stdout.

        Parameters
        ----------
        delta_message : str
            The delta message (the part that has not been streamed to stdout yet).
        """
        print(delta_message, end="", flush=True)
        post_partial(delta_message)

    def stopped_callback(self):
        r"""Stream an additional '\n' when generation ends."""
        print()
        notify_generating_thought(False)


def post_partial(chunk: str) -> bool:
    # print("Posting partial message chunk...")
    # Define the POST endpoint
    post_endpoint = f"http://{listener.server_ip}:8080/streamPartialSheepThought"
    # for _ in range(max_retries):
    try:
        requests.post(post_endpoint, json={"message": chunk})
        # print(f"Partial message chunk posted: {chunk}")
        return True
    except Exception as e:
        print(f"Error posting partial message: {e}")
    return False


def post_message(output: str) -> bool:
    global last_posted_thought
    print("Posting new thought...")
    post_endpoint = f"http://{listener.server_ip}:8080/newSheepThought"
    if output == last_posted_thought:
        print("Thought is the same as the last posted thought, skipping post.")
        return False
    for _ in range(max_retries):
        try:
            requests.post(post_endpoint, json={"message": output})
            print("Thought posted")
            last_posted_thought = output
            return True
        except Exception as e:
            print(f"Error posting message: {e}")
    return False


def notify_generating_thought(generating: bool) -> bool:
    print(f"Notifying sheep is thinking={generating}...")
    post_endpoint = f"http://{listener.server_ip}:8080/isGeneratingThought"
    for _ in range(max_retries):
        try:
            requests.post(post_endpoint, json={"isGenerating": generating})
            print("Sheep is thinking notification posted")
            return True
        except Exception as e:
            print(f"Error posting sheep is thinking notification: {e}")
    return False


def generate_response(messages: List[str]) -> Optional[str]:
    print("Generating response...")
    prompt = ""  # "Here are the most recent messages people have written: \n"
    prompt += "\n".join([msg for msg in messages])
    for _ in range(max_retries):
        try:
            if args.model == "llama2":
                response = cm.generate(
                    prompt=prompt,
                    progress_callback=ResponseCallback(callback_interval=2),
                )
                print("Response generated")
                print(cm.stats())
                print(
                    f"Current RAM usage: {psutil.Process().memory_info().rss / 1024 ** 2} MB"
                )
                return response
            else:
                response = openai.ChatCompletion.create(
                    model="gpt-4-1106-preview",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0,
                    stream=True,
                )
                callback = ResponseCallback(callback_interval=2)
                total_response = ""
                for chunk in response:
                    if chunk.choices[0].finish_reason == "stop":
                        callback.stopped_callback()
                    else:
                        callback.delta_callback(chunk.choices[0].delta.content)
                        total_response += chunk.choices[0].delta.content

                print("Response generated")
                return total_response
        except Exception as e:
            print(f"Error generating response: {e}")
            notify_generating_thought(False)
    return None


# generate_response(["hello", "how are you?"])

while True:
    messages = get_messages()
    if messages is not None:
        if len(messages) > 0:
            newmessages = "\n".join([msg for msg in messages])
            print(f"New messages\n: {newmessages}")
            output = generate_response(messages)
            if output is not None:
                post_message(output)
        else:
            print("No new messages, skipping response generation.")
    time.sleep(5)
