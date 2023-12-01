import os
import time
from typing import Any

import openai

from higgins.datasets import messaging_datasets
from higgins.nlp import nlp_utils

from . import caching
from . import completion_utils


openai.api_key = os.getenv("OPENAI_API_KEY")


def send_message_completion(cmd: str, engine="davinci", cache: Any = None):
    prompt = completion_utils.build_completion_prompt(
        question=cmd,
        action_chains=messaging_datasets.SEND_MESSAGE_DATASET_TRAIN,
        task_description="Convert the following text into commands",
    )
    # print(f"Prompt: {prompt}")
    cache = cache if cache is not None else caching.get_default_cache()
    cache_key = nlp_utils.hash_normalized_text(prompt)
    if cache_key not in cache:
        start = time.time()
        response = openai.Completion.create(
            engine=engine,
            model=None,
            prompt=prompt,
            temperature=0.2,
            max_tokens=100,
            top_p=1.0,
            frequency_penalty=0.2,
            presence_penalty=0.0,
            stop=["<<END>>"],
        )
        # print(f"Time: {time.time() - start:.2f}")
        answer = response["choices"][0]["text"].strip("Q:").strip()
        cache.add(
            key=cache_key,
            value={
                "cmd": cmd,
                "answer": answer,
                "response": response
            }
        )
    else:
        answer = cache[cache_key]["answer"]
        response = cache[cache_key]["response"]

    return answer


def test_send_message_completion():
    for example in messaging_datasets.SEND_MESSAGE_DATASET_TEST:
        answer = send_message_completion(example["query"])
        actions = completion_utils.convert_string_to_action_chain(answer)
        print(f"Q: {example['query']}\nA: {answer}\nI: {actions}\nE: {example['actions']}")
        assert actions == example['actions']


if __name__ == "__main__":
    test_send_message_completion()
