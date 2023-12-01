"""Runs the wikipedia article QA generation pipeline"""
import argparse
import asyncio
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from typing import List

from langchain.chat_models.openai import ChatOpenAI

from wiki_gen_qa.generate_qa import get_wiki_article_qa_facts
from wiki_gen_qa.wiki_tools import get_wikipedia_summary_sentences


async def get_wiki_article_qa(wiki_article_name: str) -> List[dict]:
    start = time.time()
    assert os.environ["OPENAI_API_KEY"] is not None
    chat = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
    summary_sentence_list = get_wikipedia_summary_sentences(wiki_article_name)
    print(f'Wikpiedia page for "{wiki_article_name}" found!')
    print(f"Generating QA for {len(summary_sentence_list)} sentences...")
    params = [
        {
            "id": ith,
            "wiki_article_name": wiki_article_name,
            "summary_sentence": summary_sentence,
            "chat": chat,
        }
        for ith, summary_sentence in enumerate(summary_sentence_list)
    ]
    # params = [(ith, wiki_article_name, summary_sentence, chat)
    #            for ith, summary_sentence in enumerate(summary_sentence_list)]
    with ThreadPoolExecutor(max_workers=8) as executor:
        loop = asyncio.get_running_loop()
        tasks = [
            loop.run_in_executor(executor, partial(get_wiki_article_qa_facts, **param))
            for param in params
        ]
        result_data = await asyncio.gather(*tasks)
    print("finished in ", int(time.time() - start))
    return result_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process a Wikipedia article")
    parser.add_argument(
        "wiki_article_name", type=str, help="Name of the Wikipedia article"
    )
    parser.add_argument(
        "-o", type=str, default="generated_qa.json", help="Path to output JSON file"
    )
    args = parser.parse_args()
    generated_data = asyncio.run(get_wiki_article_qa(args.wiki_article_name))
    with open(args.o, "w", encoding="utf-8") as f:
        json.dump(generated_data, f, indent=4)
    print(
        f"Successfully generated QA for {args.wiki_article_name} and saved to {args.o}!"
    )
