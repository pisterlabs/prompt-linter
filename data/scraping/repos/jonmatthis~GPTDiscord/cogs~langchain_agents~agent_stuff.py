import io
import json
import re
import sys
import tempfile
import traceback
from typing import Optional, Dict, Any

import aiohttp
from bs4 import BeautifulSoup
from langchain.agents.conversational_chat.output_parser import ConvoOutputParser
from langchain.requests import Requests
from llama_index import SimpleDirectoryReader, OpenAIEmbedding, ServiceContext, GPTSimpleVectorIndex
from llama_index.prompts.chat_prompts import CHAT_REFINE_PROMPT
from pydantic import BaseModel, Extra
from transformers import GPT2TokenizerFast


def my_parse(self, text):
    # Remove all pairs of triple backticks from the input. However, don't remove pairs of ```json and ```. Only remove ``` and ``` pairs, maintain the text between the pairs so that only the backticks
    # are removed and the text is left intact.
    text_without_triple_backticks = re.sub(
        r"```(?!json)(.*?)```", r"\1", text, flags=re.DOTALL
    )

    # Call the original parse() method with the modified input
    try:
        result = original_parse(self, text_without_triple_backticks)
    except Exception:
        traceback.print_exc()
        # Take the text and format it like
        # {
        #     "action": "Final Answer",
        #     "action_input": text
        # }
        # This will cause the bot to respond with the text as if it were a final answer.
        if "action_input" not in text_without_triple_backticks:
            text_without_triple_backticks = f'{{"action": "Final Answer", "action_input": {json.dumps(text_without_triple_backticks)}}}'
            result = original_parse(self, text_without_triple_backticks)

        else:
            # Insert "```json" before the opening curly brace
            text_without_triple_backticks = re.sub(
                r"({)", r"```json \1", text_without_triple_backticks
            )

            # Insert "```" after the closing curly brace
            text_without_triple_backticks = re.sub(
                r"(})", r"\1 ```", text_without_triple_backticks
            )

            result = original_parse(self, text_without_triple_backticks)

    return result


class CaptureStdout:
    def __enter__(self):
        self.buffer = io.StringIO()
        self.original_stdout = sys.stdout
        sys.stdout = self.buffer
        return self.buffer

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout = self.original_stdout


class RedoSearchUser:
    def __init__(self, ctx, query, search_scope, nodes, response_mode):
        self.ctx = ctx
        self.query = query
        self.search_scope = search_scope
        self.nodes = nodes
        self.response_mode = response_mode


class CustomTextRequestWrapper(BaseModel):
    """Lightweight wrapper around requests library.

    The main purpose of this wrapper is to always return a text output.
    """

    headers: Optional[Dict[str, str]] = None
    aiosession: Optional[aiohttp.ClientSession] = None
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid
        arbitrary_types_allowed = True

    def __init__(self, **data: Any):
        super().__init__(**data)

    @property
    def requests(self) -> Requests:
        return Requests(headers=self.headers, aiosession=self.aiosession)

    def get(self, url: str, **kwargs: Any) -> str:
        # the "url" field is actuall some input from the LLM, it is a comma separated string of the url and a boolean value and the original query
        try:
            url, use_gpt4, original_query = url.split(",")
        except:
            url = url
            use_gpt4 = False
            original_query = "No Original Query Provided"

        use_gpt4 = use_gpt4 == "True"
        """GET the URL and return the text."""
        text = self.requests.get(url, **kwargs).text

        # Load this text into BeautifulSoup, clean it up and only retain text content within <p> and <title> and <h1> type tags, get rid of all javascript and css too.
        soup = BeautifulSoup(text, "html.parser")

        # Decompose script, style, head, and meta tags
        for tag in soup(["script", "style", "head", "meta"]):
            tag.decompose()

        # Get remaining text from the soup object
        text = soup.get_text()

        # Clean up white spaces
        text = re.sub(r"\s+", " ", text)

        # If not using GPT-4 and the text token amount is over 3500, truncate it to 3500 tokens
        tokens = len(self.tokenizer(text)["input_ids"])
        print("The scraped text content is: " + text)
        if len(text) < 5:
            return "This website could not be scraped. I cannot answer this question."
        if (not use_gpt4 and tokens > 3000) or (use_gpt4 and tokens > 7000):
            with tempfile.NamedTemporaryFile(mode="w", delete=False) as f:
                f.write(text)
                f.close()
                document = SimpleDirectoryReader(input_files=[f.name]).load_data()
                embed_model = OpenAIEmbedding()
                service_context = ServiceContext.from_defaults(embed_model=embed_model)
                index = GPTSimpleVectorIndex.from_documents(
                    document, service_context=service_context, use_async=True
                )
                response_text = index.query(
                    original_query,
                    refine_template=CHAT_REFINE_PROMPT,
                    similarity_top_k=4,
                    response_mode="compact",
                )
                return response_text

        return text


original_parse = ConvoOutputParser.parse


async def capture_stdout(func, *args, **kwargs):
    with CaptureStdout() as buffer:
        result = await func(*args, **kwargs)
    captured_output = buffer.getvalue()
    return result, captured_output
