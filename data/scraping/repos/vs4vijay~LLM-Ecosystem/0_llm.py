#!/usr/bin/env python
# -*- coding: utf-8 -*-

import openai

from config import config


def system_prompt(name: str) -> str:
    return f"""
        You're a helpful assistant "{name}", here to answer the user's qeury. If you don't know
        the answer, just say "I don't know".
    """


def main():
    print(f"[+] Start program with LLM model: {config.openai_model_name}")

    openai.api_type = config.openai_api_type
    openai.api_version = config.openai_api_version
    openai.api_base = config.openai_api_base
    openai.api_key = config.openai_api_key

    user_input = input("[+] Enter your input: ")

    response = openai.ChatCompletion.create(
        model=config.openai_model_name,
        deployment_id=config.openai_deployment_name,
        # prompt=build_prompt(user_input),
        messages=[
            {"role": "system", "content": system_prompt("MS-BOT")},
            {"role": "user", "content": user_input},
        ],
        temperature=0.6,
        max_tokens=500,
    )

    print(f"[+] Response: {response}")


if __name__ == "__main__":
    main()
