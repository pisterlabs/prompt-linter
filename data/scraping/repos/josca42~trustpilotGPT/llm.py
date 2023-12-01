import tiktoken
import openai
from typing import Union, Optional
import wandb
from wandb.sdk.data_types.trace_tree import Trace
from datetime import datetime
import cohere
from assistant.config import config
import streamlit
import os

os.environ[
    "WANDB_MODE"
] = "disabled"  # FIXME: Uncomment this line to disable wandb during development

wandb.login(
    key=config["WANDB_API_KEY"],
)
run = wandb.init(
    project="trustpilot-chat",
)

LLM_cohere = cohere.Client(config["COHERE_API_KEY"])


def embed(texts: Union[list[str], str], model="cohere"):
    if isinstance(texts, str):
        texts = [texts]
    texts = [text.replace("\n", " ") for text in texts]

    if model == "cohere":
        response = LLM_cohere.embed(
            texts=texts,
            model="embed-multilingual-v2.0",
        )
        embeddings = response.embeddings
    else:
        response = openai.Embedding.create(
            input=texts,
            model="text-embedding-ada-002",
        )
        embeddings = [data.get("embedding") for data in response.data]

    return embeddings


class GPT:
    def __init__(self, log: bool, question: str, st: streamlit = None) -> None:
        self.question = question
        self.log = log
        self.root_span = None
        self.st = st

        if log:
            self.root_span = Trace(
                name="Chat",
                kind="agent",
                start_time_ms=timestamp(),
                metadata={"user": "josca"},
                inputs={"user": question},
            )

    def completion(
        self,
        messages,
        model="gpt-4",  # "gpt-3.5-turbo-0613",
        temperature=0,
        functions=[],
        stop=None,
        name="",
        kind="",
        write_to_streamlit=True,
    ) -> str:
        start = timestamp()

        stream = True if self.st and write_to_streamlit else False
        response = openai.ChatCompletion.create(
            model=model,
            messages=messages,
            temperature=temperature,
            stop=stop,
            stream=stream,
        )

        if stream:
            with self.st.chat_message("assistant", avatar="🤖"):
                message_placeholder = self.st.empty()
                full_response = ""
                for chunk in response:
                    full_response += chunk.choices[0].delta.get("content", "")
                    message_placeholder.markdown(full_response + "▌")
                message_placeholder.markdown(full_response)

        else:
            full_response = response.choices[0].message.content

        if self.log:
            self.root_span.add_child(
                Trace(
                    name=name,
                    kind=kind if kind else None,
                    start_time_ms=start,
                    end_time_ms=timestamp(),
                    inputs=wandb_format_msgs(messages),
                    outputs={"assistant": full_response},
                    model_dict={"model": model},
                )
            )

        return full_response

    def finish(self):
        self.root_span.end_time_ms = timestamp()
        self.root_span.log("chat_test")


def timestamp():
    return round(datetime.now().timestamp() * 1000)


def wandb_format_msgs(msgs):
    return {msg["role"]: msg["content"] for msg in msgs}


def num_tokens_from_string(string: str, model: str = "gpt-3.5-turbo") -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.encoding_for_model(model)
    num_tokens = len(encoding.encode(string))
    return num_tokens
