import argparse
import asyncio

import os
import openai

from utils import build_index, build_messages


# https://github.com/huggingface/transformers/issues/5486
os.environ["TOKENIZERS_PARALLELISM"] = "false"


async def main(filename: str, user_question: str):
    collection = build_index(filename)

    # Search for data given the user's question
    search_result = collection.query(
        query_texts=[user_question],
        n_results=5
    )
    context = ['\n'.join(doclist) for doclist in search_result['documents']]

    # Make a request to OpenAI chat completions API
    stream = openai.chat.completions.create(
        model="gpt-3.5-turbo-1106",
        messages=build_messages(user_question, context),
        temperature=0.25,
        top_p=0.2,
        max_tokens=512,
        stream=True
    )

    # Stream the LLM's output to screen, token by token
    for txt in stream:
        content = txt.choices[0].delta.content or ""
        print(
            content,
            end="",
            flush=True
        )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(exit_on_error=True)
    parser.add_argument('--file', type=str, required=True)
    parser.add_argument('--question', type=str, required=True)
    args = parser.parse_args()

    asyncio.run(main(args.file, args.question))
