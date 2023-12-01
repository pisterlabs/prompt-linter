import openai
import os
import threading
import time
import anthropic


def hash_messages(model, messages):
    return hash(
        model + "::" + "\n".join([m["role"] + ": " + m["content"] for m in messages])
    )


class LLM:
    def __init__(self):
        has_openai = "OPENAI_ORG" in os.environ and "OPENAI_API_KEY" in os.environ
        has_anthropic = "CLAUDE_API_KEY" in os.environ

        if not (has_openai or has_anthropic):
            raise Exception(
                "Missing OPENAI_ORG & OPENAI_API_KEY or CLAUDE_API_KEY environment variable."
            )
        self.claude_api_key = os.getenv("CLAUDE_API_KEY", None)
        openai.organization = os.getenv("OPENAI_ORG","")
        openai.api_key = os.getenv("OPENAI_API_KEY","")
        self.call_fns = {
            "chatgpt": self.call_chatgpt,
            "gpt4": self.call_gpt4,
            "claude 2": self.call_claude,
        }
        self._cached_replies = {}

    def call_claude(self, messages, key, callback):
        prompt = "\n".join([m["content"] for m in messages])
        claude = anthropic.Anthropic(api_key=self.claude_api_key)
        completion = claude.completions.create(
            model="claude-2",
            max_tokens_to_sample=512,
            prompt=f"{anthropic.HUMAN_PROMPT} {prompt}{anthropic.AI_PROMPT}",
        )
        completion = completion.completion.strip()
        if completion.startswith("Here "):
            colon_idx = completion.find(":")
            if colon_idx != -1:
                completion = completion[colon_idx + 1 :].strip()
        self._cached_replies[key] = completion
        callback(completion)

    def call_gpt4(self, messages, key, callback):
        response = openai.ChatCompletion.create(model="gpt-4", messages=messages)
        reply = response["choices"][0]["message"]["content"]
        self._cached_replies[key] = reply
        callback(reply)

    def call_chatgpt(self, messages, key, callback):
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo", messages=messages
        )
        reply = response["choices"][0]["message"]["content"]
        self._cached_replies[key] = reply
        callback(reply)
        # time.sleep(5)
        # style = messages[-2]["content"]
        # content = messages[-1]["content"]
        # # content = content[0 : min(len(content), 1000)]
        # print("calling fake llm")
        # callback("Placeholder text with style " + style + " and content " + content)

    def query(self, model, messages, callback):
        key = hash_messages(model, messages)
        msg = messages[-1]["content"][0 : min(80, len(messages[-1]["content"]))] + "..."
        if key in self._cached_replies:
            print(f"[LLM layer]: Found request ({msg}) in cache")
            callback(self._cached_replies[key])
        else:
            print("[LLM layer]: New message, query LLM in background thread")
            # style = messages[-2]["content"]
            # content = messages[-1]["content"]
            # # content = content[0 : min(len(content), 1000)]
            # print("calling fake llm")
            # callback("Placeholder text with style " + style + " and content " + content)
            #
            # self.call_llm(messages, key, callback)

            thread = threading.Thread(
                target=self.call_fns[model],
                args=(messages, key, callback),
            )
            thread.start()