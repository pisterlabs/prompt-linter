import argparse
from langchain.document_loaders import AsyncChromiumLoader, AsyncHtmlLoader
from langchain.document_transformers import (
    BeautifulSoupTransformer,
    Html2TextTransformer,
)
from managers.web_scraper import WebScraper
from managers.openai_manager import OpenAIManager
import json
import tiktoken

"""
python web_content_summary.py --urls_file "urls.txt" --result_file "result.txt" --chat_history_file "chat_history.txt" --summary_file "summary.txt" --max_encoding_length 12000
"""


class Config:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class ChatManager:
    def __init__(self, manager, chat_history_file):
        self.manager = manager
        self.chat_history_file = chat_history_file
        self.messages = []

    def add_message(self, role, content):
        message = {"role": role, "content": content}
        self.messages.append(message)
        self.save_chat_history()  # 保存对话历史

    def get_response(self):
        response = self.manager.openai_chat(self.messages)
        return response

    def save_chat_history(self):
        with open(self.chat_history_file, "a") as f:  # 使用追加模式
            f.write(
                json.dumps(self.messages[-1], ensure_ascii=False) + "\n"
            )  # 只保存最新的消息


class WebScraperManager:
    def __init__(self, config, loader, transformer):
        self.config = config
        self.scraper = WebScraper(loader, transformer)
        self.openai_manager = OpenAIManager(
            self.config.openai_api_key,
            self.config.openai_api_base,
            self.config.embed_model,
            self.config.chat_model,
        )

    def scrape_web(self, tags_to_extract=None):
        result = self.scraper.scrape(tags_to_extract)
        return result

    def openai_chat(self, messages):
        response = self.openai_manager.generate_chat_completion(
            messages,
            max_tokens=self.config.max_tokens,
            temperature=self.config.temperature,
        )
        return response


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--urls_file",
        default="urls.txt",
        help="The file containing the URLs of the webpages to be scraped.",
    )
    parser.add_argument(
        "--result_file",
        default="result.txt",
        help="The file in which to store the result of the scraping.",
    )
    parser.add_argument(
        "--chat_history_file",
        default="chat_history.txt",
        help="The file in which to store the chat history.",
    )
    parser.add_argument(
        "--summary_file",
        default="summary.txt",
        help="The file in which to store the summary.",
    )
    parser.add_argument(
        "--max_encoding_length",
        default=12000,
        type=int,
        help="The maximum length for encoding.",
    )
    parser.add_argument(
        "--openai_api_key",
        default="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",
        help="The API key for OpenAI.",
    )
    parser.add_argument(
        "--openai_api_base",
        default="https://api.openai.com/v1",
        help="The base URL for the OpenAI API.",
    )
    parser.add_argument(
        "--embed_model",
        default="text-embedding-ada-002",
        help="The model to use for embedding.",
    )
    parser.add_argument(
        "--chat_model",
        default="gpt-3.5-turbo-16k-0613",
        help="The model to use for chat.",
    )
    parser.add_argument(
        "--max_tokens",
        default=1500,
        type=int,
        help="The maximum number of tokens for the generated response.",
    )
    parser.add_argument(
        "--temperature",
        default=0.1,
        type=float,
        help="The temperature to use for the generated response.",
    )
    args = parser.parse_args()

    config = Config(
        openai_api_key=args.openai_api_key,
        openai_api_base=args.openai_api_base,
        embed_model=args.embed_model,
        chat_model=args.chat_model,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
    )
    encoding = tiktoken.encoding_for_model(args.chat_model)

    with open(args.urls_file, 'r') as urls_file:
        urls = [url.strip() for url in urls_file.readlines()]
    print(urls)

    with open(args.summary_file, 'w') as summary_file:
        for url in urls:
            manager = WebScraperManager(
                config,
                AsyncHtmlLoader([url]),
                Html2TextTransformer(),
            )

            result = manager.scrape_web()[0].page_content
            if len(encoding.encode(result)) >= args.max_encoding_length:
                continue

            with open(args.result_file, "w") as result_file:
                result_file.write(result)

            chat_manager = ChatManager(manager, args.chat_history_file)

            chat_manager.add_message("system", "You are a helpful assistant.")
            chat_manager.add_message(
                "user",
                "我想知道下面这篇文章主要讲了什么，提到了哪些概念相关描述，哪些功能性描述。\n" + result,
            )

            response = chat_manager.get_response()
            response_content = response["choices"][0]["message"]["content"]
            total_tokens = response["usage"]["total_tokens"]
            prompt_tokens = response["usage"]["prompt_tokens"]
            completion_tokens = response["usage"]["completion_tokens"]
            print(
                f"Consume tokens: {total_tokens}, Prompt tokens: {prompt_tokens}, Completion tokens: {completion_tokens}"
            )
            chat_manager.add_message("assistant", response_content)
            print(response_content)
            summary_file.write('url:\n'+url+'\n'+'summary:\n'+response_content+'\n\n\n')


if __name__ == "__main__":
    main()
