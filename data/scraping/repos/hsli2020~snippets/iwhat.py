# https://github.com/yihong0618/iWhat/blob/main/what/what.py

import string
from rich import print
from rich.table import Table

import openai


class What:
    def __init__(self, what, is_en=False, api_base=None):
        if api_base:
            openai.api_base = api_base
        self.what = what
        self.what_prompt = (
            "f这个 `{what}` 可能是什么，请分行回答，第一行回答他最可能是的东西（要求精确些），第二行回答这个东西的描述".format(
                what=what
            )
        )
        if is_en:
            self.what_prompt = self.what_prompt + ", 请用英语回答"

    def _to_what(self):
        completion = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": self.what_prompt}],
        )
        return completion["choices"][0]["message"]["content"]

    @staticmethod
    def _is_all_punctuations(s):
        return all(c in string.punctuation for c in s)

    def show_what(self):
        what = self._to_what()
        index = 0
        maybe = None
        trim_what = [w for w in what.splitlines() if not self._is_all_punctuations(w)]
        if not trim_what:
            raise Exception("No what!")
        maybe = trim_what[0]
        desc = "\n".join(trim_what[1:])
        desc = desc.replace("，", "，\r\n")
        desc = desc.replace("。", "。\r\n")
        desc = desc[:-2]

        table = Table(title="What is it AI")
        table.add_column("What", style="cyan")
        table.add_column("Maybe", style="red", justify="middle")
        table.add_column("Desc", justify="left", style="green")
        table.add_row(self.what, maybe, desc)
        # print(what)
        print(table)

# https://raw.githubusercontent.com/yihong0618/iWhat/main/what/cli.py

from what.what import What

import argparse
import os
from os import environ as env


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("what", help="what is it")
    parser.add_argument(
        "--openai_key", dest="openai_key", type=str, default="", help="OpenAI api key"
    )
    parser.add_argument(
        "-p",
        "--proxy",
        dest="proxy",
        type=str,
        default="",
        help="use proxy like http://127.0.0.1:7890",
    )
    # args to change api_base
    parser.add_argument(
        "--api_base",
        dest="api_base",
        type=str,
        help="specify base url other than the OpenAI's official API address",
    )
    parser.add_argument(
        "--en", dest="en", action="store_true", help="If use English to answer"
    )

    options = parser.parse_args()
    PROXY = options.proxy
    if PROXY != "":
        os.environ["http_proxy"] = PROXY
        os.environ["https_proxy"] = PROXY

    OPENAI_API_KEY = options.openai_key or env.get("OPENAI_API_KEY")
    if not OPENAI_API_KEY:
        raise Exception("OpenAI API key not provided, please google how to obtain it")

    what = What(options.what, is_en=options.en, api_base=options.api_base)
    what.show_what()


if __name__ == "__main__":
    main()
