"""
Module containing AsyncBaseReader class for reading and summarizing papers with chatbot assistance.
"""
import asyncio
import base64
import datetime
import re
from pathlib import Path
from typing import List

import openai
import requests
import tenacity
import tiktoken
from loguru import logger

from .aexport import aexport
from .paper_with_image import Paper
from .utils import load_config


class AsyncBaseReader:
    """
    Class for reading and summarizing papers with chatbot assistance asynchronously.
    """

    def __init__(
        self, root_path: str, language: str, file_format: str, save_image: bool
    ):
        """
        Initializes AsyncBaseReader object with root path, language, file format, and save image flag.

        Args:
            root_path (str): Root path for saving files.
            language (str): Language to use for chatbot.
            file_format (str): File format to save papers in.
            save_image (bool): Flag indicating whether to save images of papers.
        """
        if isinstance(root_path, str):
            root_path = Path(root_path)

        self.root_path = root_path
        self.language = language
        self.file_format = file_format

        self.config, self.chat_api_list = load_config()
        self.cur_api = 0

        self.gitee_key = self.config["Gitee"]["api"] if save_image else ""

        self.max_token_num = 4096
        self.encoding = tiktoken.get_encoding("gpt2")
        self.token_usage = 0

    async def _summary_with_chat(self, paper_list: List[Paper], key_words: List[str]):
        """
        Asynchronously summarizes papers with chatbot assistance.

        Args:
            paper_list (List[Paper]): List of Paper objects to summarize.
            key_words (List[str]): List of key words to use for chatbot.

        Returns:
            None
        """
        await asyncio.gather(
            *[
                self.summary_with_chat_for_one_paper(paper, index, key_words)
                for index, paper in enumerate(paper_list)
            ]
        )

    def summary_with_chat(self, paper_list: List[Paper], key_words: List[str]):
        """
        Summarizes papers with chatbot assistance.

        Args:
            paper_list (List[Paper]): List of Paper objects to summarize.
            key_words (List[str]): List of key words to use for chatbot.

        Returns:
            None
        """
        asyncio.run(self._summary_with_chat(paper_list, key_words))

    def update_title(self, text: str) -> str:
        """
        Updates the title of a paper based on the summary text.

        Args:
            text (str): Summary text to extract title from.

        Returns:
            str: Updated title.
        """
        for line in text.split("\n"):
            if "Title:" in line:
                return line.split("Title:")[1].strip()

    async def summary_with_chat_for_one_paper(
        self, paper: Paper, paper_index: int, key_words
    ):
        """
        Asynchronously summarizes a single paper with chatbot assistance.

        Args:
            paper (Paper): Paper object to summarize.
            paper_index (int): Index of the paper in the list.
            key_words (List[str]): List of key words to use for chatbot.

        Returns:
            None
        """

        result = []
        text = ""
        text += "Title:" + paper.title
        text += "Url:" + paper.url
        text += "Abstrat:" + paper.abs
        text += "Paper_info:" + paper.sections["paper_info"].text
        # abstract
        text += list(paper.sections.sections())[0].text
        logger.trace(f"summary paper_info: {text}")
        chat_summary_text = ""

        try:
            chat_summary_text = await self.chat_summary(text=text, key_words=key_words)
        except Exception as e:
            logger.warning(f"summary_error: {e}")
            if "maximum context" in str(e):
                current_tokens_index = (
                    str(e).find("your messages resulted in")
                    + len("your messages resulted in")
                    + 1
                )
                offset = int(str(e)[current_tokens_index : current_tokens_index + 4])
                summary_prompt_token = offset + 1000 + 150
                chat_summary_text = await self.chat_summary(
                    text=text,
                    key_words=key_words,
                    summary_prompt_token=summary_prompt_token,
                )
            else:
                raise e

        if (
            paper.title == ""
            and (title := self.update_title(chat_summary_text)) is not None
        ):
            paper.title = title

        result.append("## Paper:" + str(paper_index + 1))
        result.append("\n\n\n")
        result.append(chat_summary_text)

        method_section = None
        method_sections = paper.sections.get_method()
        for section in method_sections:
            if section.has_text():
                method_section = section

        if method_section is not None:
            text = ""
            method_text = ""
            summary_text = ""
            summary_text += "<summary>" + chat_summary_text
            # methods
            method_text += method_section.text
            logger.trace(f"method_text: {method_text}")
            text = summary_text + "\n\n<Methods>:\n\n" + method_text

            chat_method_text = ""
            try:
                chat_method_text = await self.chat_method(
                    text=text, key_words=key_words
                )
            except Exception as e:
                logger.error(f"method_error: {e}")
                if "maximum context" in str(e):
                    current_tokens_index = (
                        str(e).find("your messages resulted in")
                        + len("your messages resulted in")
                        + 1
                    )
                    offset = int(
                        str(e)[current_tokens_index : current_tokens_index + 4]
                    )
                    method_prompt_token = offset + 800 + 150
                    chat_method_text = await self.chat_method(
                        text=text,
                        key_words=key_words,
                        method_prompt_token=method_prompt_token,
                    )
            result.append(chat_method_text)
        else:
            chat_method_text = ""

        result.append("\n" * 4)

        # 第三步总结全文，并打分：
        conclusion_section = None
        for section in paper.sections.get_conclusion():
            if section.has_text():
                conclusion_section = section
                break

        text = ""
        conclusion_text = ""
        summary_text = ""
        summary_text += (
            "<summary>"
            + chat_summary_text
            + "\n <Method summary>:\n"
            + chat_method_text
        )

        if conclusion_section is not None:
            conclusion_text += conclusion_section.text
            text = summary_text + "\n\n<Conclusion>:\n\n" + conclusion_text
        else:
            text = summary_text

        chat_conclusion_text = ""
        try:
            chat_conclusion_text = await self.chat_conclusion(
                text=text, key_words=key_words
            )
        except Exception as e:
            logger.info(f"conclusion_error: {e}")
            if "maximum context" in str(e):
                current_tokens_index = (
                    str(e).find("your messages resulted in")
                    + len("your messages resulted in")
                    + 1
                )
                offset = int(str(e)[current_tokens_index : current_tokens_index + 4])
                conclusion_prompt_token = offset + 800 + 150
                chat_conclusion_text = await self.chat_conclusion(
                    text=text,
                    key_words=key_words,
                    conclusion_prompt_token=conclusion_prompt_token,
                )
        result.append(chat_conclusion_text)
        result.append("\n" * 4)

        date_str = str(datetime.datetime.now())[:13].replace(" ", "-")
        export_path = self.root_path / "export"

        if not export_path.exists():
            export_path.mkdir(parents=True, exist_ok=True)

        file_name = (
            Path(export_path)
            / f"{date_str}-{self.validateTitle(paper.title[:80])}".strip()
        )

        await aexport(
            content="\n".join([item.strip() for item in result]),
            file_name=file_name.with_suffix(f".{self.file_format}"),
        )

    @tenacity.retry(
        wait=tenacity.wait_exponential(multiplier=1, min=4, max=10),
        stop=tenacity.stop_after_attempt(5),
        reraise=True,
    )
    async def chat_conclusion(
        self, text, key_words, conclusion_prompt_token=800
    ) -> str:
        """
        Generates a conclusion for a given text using OpenAI's GPT-3 API.

        Args:
            text (str): The text to generate a conclusion for.
            key_words (str): The key words related to the text.
            conclusion_prompt_token (int, optional): The number of tokens to use as a prompt for the conclusion. Defaults to 800.

        Returns:
            str: The generated conclusion.
        """

        openai.api_key = self.chat_api_list[self.cur_api]
        self.cur_api += 1
        self.cur_api = (
            0 if self.cur_api >= len(self.chat_api_list) - 1 else self.cur_api
        )
        text_token = len(self.encoding.encode(text))
        clip_text_index = int(
            len(text) * (self.max_token_num - conclusion_prompt_token) / text_token
        )
        clip_text = text[:clip_text_index]

        messages = [
            {
                "role": "system",
                "content": f"You are a reviewer in the field of [{key_words}] and you need to critically review this article",
            },
            {
                "role": "assistant",
                "content": "This is the <summary> and <conclusion> part of an English literature, where <summary> you have already summarized, but <conclusion> part, I need your help to summarize the following questions:"
                + clip_text,
            },
            {
                "role": "user",
                "content": """
                 9. Make the following summary.Be sure to use {} answers (proper nouns need to be marked in English).
                    - (1):What is the significance of this piece of work?
                    - (2):Summarize the strengths and weaknesses of this article in three dimensions: innovation point, performance, and workload.
                    .......
                 Follow the format of the output later:
                 9. Conclusion: \n\n
                    - (1):xxx;\n
                    - (2):Innovation point: xxx; Performance: xxx; Workload: xxx;\n

                 Be sure to use {} answers (proper nouns need to be marked in English), statements as concise and academic as possible, do not repeat the content of the previous <summary>, the value of the use of the original numbers, be sure to strictly follow the format, the corresponding content output to xxx, in accordance with \n line feed, ....... means fill in according to the actual requirements, if not, you can not write.
                 """.format(
                    self.language, self.language
                ),
            },
        ]
        response = await openai.ChatCompletion.acreate(
            model="gpt-3.5-turbo",
            # prompt需要用英语替换，少占用token。
            messages=messages,
        )
        result = ""
        for choice in response.choices:
            result += choice.message.content

        result = self.format_text(result)
        logger.trace(f"conclusion_result:\n{result}")
        self.report_token_usage(response)

        return result

    @tenacity.retry(
        wait=tenacity.wait_exponential(multiplier=1, min=4, max=10),
        stop=tenacity.stop_after_attempt(5),
        reraise=True,
    )
    async def chat_method(self, text, key_words, method_prompt_token=800) -> str:
        """
        This function is responsible for generating a chat response to a given text prompt and key words. It uses OpenAI's GPT-3 model to generate a response.

        The function takes in the following parameters:
        - text: a string representing the text prompt to generate a response to
        - key_words: a string representing the key words related to the text prompt
        - method_prompt_token: an integer representing the maximum number of tokens to use for the method prompt

        The function returns a string representing the generated response.

        The function uses the tenacity library to retry the chat method up to 5 times if it fails.
        The chat_method function is decorated with the @tenacity.retry decorator, which specifies the retry behavior.
        The function first sets the OpenAI API key and then creates a list of messages to send to the GPT-3 model. The messages include a system message, an assistant message, and a user message.
        The system message informs the user that they are a researcher in the field of the given key words.
        The assistant message provides context for the user and includes a clipped version of the text prompt.
        The user message includes a set of instructions for the user to follow in order to generate a response.
        The function then uses the OpenAI ChatCompletion API to generate a response based on the messages sent to the model.
        The response is then formatted and returned as the output of the function."""

        openai.api_key = self.chat_api_list[self.cur_api]
        self.cur_api += 1
        self.cur_api = (
            0 if self.cur_api >= len(self.chat_api_list) - 1 else self.cur_api
        )
        text_token = len(self.encoding.encode(text))
        clip_text_index = int(
            len(text) * (self.max_token_num - method_prompt_token) / text_token
        )
        clip_text = text[:clip_text_index]

        messages = [
            {
                "role": "system",
                "content": f"You are a researcher in the field of [{key_words}] who is good at summarizing papers using concise statements",
            },
            {
                "role": "assistant",
                "content": "This is the <summary> and <Method> part of an English document, where <summary> you have summarized, but the <Methods> part, I need your help to read and summarize the following questions."
                + clip_text,
            },
            {
                "role": "user",
                "content": """
                 8. Describe in detail the methodological idea of this article. Be sure to use {} answers (proper nouns need to be marked in English). For example, its steps are.
                    - (1):...
                    - (2):...
                    - (3):...
                    - .......
                 Follow the format of the output that follows:
                 8. Methods: \n\n
                    - (1):xxx;\n
                    - (2):xxx;\n
                    - (3):xxx;\n
                    ....... \n\n

                 Be sure to use {} answers (proper nouns need to be marked in English), statements as concise and academic as possible, do not repeat the content of the previous <summary>, the value of the use of the original numbers, be sure to strictly follow the format, the corresponding content output to xxx, in accordance with \n line feed, ....... means fill in according to the actual requirements, if not, you can not write.
                 """.format(
                    self.language, self.language
                ),
            },
        ]
        response = await openai.ChatCompletion.acreate(
            model="gpt-3.5-turbo",
            messages=messages,
        )

        result = ""
        for choice in response.choices:
            result += choice.message.content

        result = self.format_text(result)
        logger.trace(f"method_result:\n{result}")
        self.report_token_usage(response)

        return result

    @tenacity.retry(
        wait=tenacity.wait_exponential(multiplier=1, min=4, max=10),
        stop=tenacity.stop_after_attempt(3),
        reraise=True,
    )
    async def chat_summary(self, text, key_words, summary_prompt_token=1100) -> str:
        """
        Summarizes a research paper based on a set of instructions provided to the user.

        Args:
            text (str): The text of the research paper to be summarized.
            key_words (str): The keywords associated with the research paper.
            summary_prompt_token (int): The maximum number of tokens to be used in the summary prompt.

        Returns:
            str: The summarized text of the research paper.
        """
        openai.api_key = self.chat_api_list[self.cur_api]
        self.cur_api += 1
        self.cur_api = (
            0 if self.cur_api >= len(self.chat_api_list) - 1 else self.cur_api
        )
        text_token = len(self.encoding.encode(text))
        clip_text_index = int(
            len(text) * (self.max_token_num - summary_prompt_token) / text_token
        )
        clip_text = text[:clip_text_index]

        messages = [
            {
                "role": "system",
                "content": f"You are a researcher in the field of [{key_words}]  who is good at summarizing papers using concise statements",
            },
            {
                "role": "assistant",
                "content": "This is the title, author, link, abstract and introduction of an English document. I need your help to read and summarize the following questions: "
                + clip_text,
            },
            {
                "role": "user",
                "content": f"""
                 1. Mark the title of the paper
                 2. list all the authors' names (use English)
                 3. mark the first author's affiliation (use English)
                 4. mark the keywords of this article (use English)
                 5. mark the link to the paper
                 6. mark Github code link (if available, fill in Github:None if not)
                 7. summarize according to the following four points.Be sure to use {self.language} answers (proper nouns need to be marked in English)
                    - (1):What is the research background of this article?
                    - (2):What are the past methods? What are the problems with them? Is the approach well motivated?
                    - (3):What is the research methodology proposed in this paper?
                    - (4):On what task and what performance is achieved by the methods in this paper? Can the performance support their goals?
                 Follow the format of the output that follows:
                 1. Title: xxx\n\n
                 2. Authors: xxx\n\n
                 3. Affiliation: xxx\n\n
                 4. Keywords: xxx\n\n
                 5. Urls: xxx or xxx , xxx \n\n
                 6. Github: xxx or xxx , xxx \n\n
                 7. Summary: \n\n
                    - (1):xxx;\n
                    - (2):xxx;\n
                    - (3):xxx;\n
                    - (4):xxx.\n\n

                 Be sure to use {self.language} answers (proper nouns need to be marked in English), statements as concise and academic as possible, do not have too much repetitive information, numerical values using the original numbers, be sure to strictly follow the format,
                 the corresponding content output to xxx, in accordance with \n line feed.
                 """,
            },
        ]

        response = await openai.ChatCompletion.acreate(
            model="gpt-3.5-turbo",
            messages=messages,
        )
        result = ""

        for choice in response.choices:
            result += choice.message.content

        result = self.format_text(result)
        logger.trace(f"summary_result:\n{result}")
        self.report_token_usage(response)
        return result

    @staticmethod
    def format_text(text: str) -> str:
        """
        Formats the text by removing unnecessary whitespaces and newlines.
        Args:
            text (str): The text to be formatted.
        Returns:
            str: The formatted text.
        """
        result = ""
        for line in text.split("\n"):
            result += line.strip() + "\n"
        return result

        result = ""
        for line in text.split("\n"):
            result += line.strip() + "\n"
        return result

    def validateTitle(self, title: str) -> str:
        """
        Validates the title by removing any invalid characters and replacing them with underscores.
        Args:
            title (str): The title to be validated.
        Returns:
            str: The validated title.
        """
        rstr = r"[\/\\\:\*\?\"\<\>\|]"  # '/ \ : * ? " < > |'
        new_title = re.sub(rstr, "_", title)  # 替换为下划线
        return new_title

    def show_token_usage(self):
        """
        Displays the current token usage and corresponding price in USD.
        """
        money = self.token_usage / 1000 * 0.002
        logger.info(f"TOKENS: {self.token_usage} / PRICES: ${money:.6f}")

    def report_token_usage(self, response):
        """
        Reports the token usage and response time for a given response.
        Args:
            response: The response object returned by the OpenAI API.
        """
        logger.trace(f"prompt_token_used: {response.usage.prompt_tokens}")
        logger.trace(f"completion_token_used: {response.usage.completion_tokens}")
        logger.trace(f"total_token_used: {response.usage.total_tokens}")
        logger.trace(f"response_time: { response.response_ms / 1000.0}s")
        self.token_usage += response.usage.total_tokens

    @tenacity.retry(
        wait=tenacity.wait_exponential(multiplier=1, min=4, max=10),
        stop=tenacity.stop_after_attempt(5),
        reraise=True,
    )
    def upload_gitee(self, image_path, image_name="", ext="png"):
        """
        上传到码云
        :return:
        """
        with open(image_path, "rb") as f:
            base64_data = base64.b64encode(f.read())
            base64_content = base64_data.decode()

        date_str = (
            str(datetime.datetime.now())[:19].replace(":", "-").replace(" ", "-")
            + "."
            + ext
        )
        path = image_name + "-" + date_str

        payload = {
            "access_token": self.gitee_key,
            "owner": self.config["Gitee"]["owner"],
            "repo": self.config["Gitee"]["repo"],
            "path": self.config["Gitee"]["path"],
            "content": base64_content,
            "message": "upload image",
        }
        # 这里需要修改成你的gitee的账户和仓库名，以及文件夹的名字：
        url = (
            "https://gitee.com/api/v5/repos/"
            + self.config["Gitee"]["owner"]
            + "/"
            + self.config["Gitee"]["repo"]
            + "/contents/"
            + self.config["Gitee"]["path"]
            + "/"
            + path
        )
        rep = requests.post(url, json=payload).json()
        logger.info(f"{rep=}")
        if "content" in rep.keys():
            image_url = rep["content"]["download_url"]
        else:
            image_url = (
                r"https://gitee.com/api/v5/repos/"
                + self.config["Gitee"]["owner"]
                + "/contents/"
                + self.config["Gitee"]["path"]
                + "/"
                + path
            )

        return image_url
