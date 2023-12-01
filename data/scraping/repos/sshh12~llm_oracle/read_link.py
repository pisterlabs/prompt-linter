from typing import Optional, List

from langchain.agents import Tool

from llm_oracle.link_scraping import scrape_text, chunk_and_strip_html
from llm_oracle import llm, processing_utils

CHUNK_SIZE = 4000

SUMMARIZE_PROMPT = """
```
{chunk}
```

Using the above text from scraping a website, create a very detailed summary of the information.

Include key statistics, dates, numbers, and other details.

Note: This will be used to answer "{ask}"
"""

SUMMARIZE_CHUNKS_PROMPT = """
```
{chunks}
```

Using the above text from scraping a website, create a very detailed summary of the information and provide information about "{ask}".

Include key statistics, dates, numbers, and other details.
"""


@processing_utils.cache_func
def _summarize_chunk(model: llm.LLMModel, chunk: str, ask: str) -> str:
    return model.call_as_llm(SUMMARIZE_PROMPT.format(chunk=chunk, ask=ask))


def _summarize_chunks(model: llm.LLMModel, chunks: List[str], ask: str) -> str:
    return model.call_as_llm(SUMMARIZE_CHUNKS_PROMPT.format(chunks="\n\n".join(chunks), ask=ask))


def summarize_chunks(model: llm.LLMModel, chunks: List[str], ask: str) -> str:
    summary_chunks = []
    for chunk in chunks:
        summary_chunks.append(_summarize_chunk(model, chunk, ask))
    return _summarize_chunks(model, summary_chunks, ask)


class ReadLinkWrapper:
    def __init__(self, summary_model: Optional[llm.LLMModel] = None, use_proxy: bool = True):
        if summary_model is None:
            self.summary_model = llm.get_default_fast_llm()
        else:
            self.summary_model = summary_model
        self.use_proxy = use_proxy

    def run(self, query: str) -> str:
        if query.endswith(".pdf"):
            return "Cannot read links that end in pdf"
        try:
            url, ask = query.split(", ")
        except ValueError:
            return 'input was in the wrong format, it should be "url, question"'
        chunks = chunk_and_strip_html(scrape_text(url, use_proxy=self.use_proxy), CHUNK_SIZE)
        return summarize_chunks(self.summary_model, chunks, ask)


def get_read_link_tool(**kwargs) -> Tool:
    read_link = ReadLinkWrapper(**kwargs)
    return Tool(
        name="Read Link",
        func=read_link.run,
        description='useful to read and extract the contents of any link. the input should be "url, question", e.g. "https://en.wikipedia.org/wiki/2023_in_the_United_States, list of events in april 2023"',
    )
