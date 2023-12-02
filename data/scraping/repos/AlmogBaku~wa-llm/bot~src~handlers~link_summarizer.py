import re
from datetime import timedelta
from typing import List, Generator

from langchain.chains.summarize import load_summarize_chain
from langchain.chains.summarize.map_reduce_prompt import prompt_template
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import PlaywrightURLLoader
from langchain.schema import Document
from langchain.text_splitter import CharacterTextSplitter
from loguru import logger

from ..events import Context, CommandResult, msg_cmd, Message, message_handler
from ..jid import parse_jid
from ..store import ChatStore


def find_links(msg: str) -> list[str]:
    pattern = r"(?:(?:https?):\/\/|www\.)(?:\([-A-Z0-9+&@#\/%=~_|$?!:,.]*\)|[-A-Z0-9+&@#\/%=~_|$?!:,.])*" \
              r"(?:\([-A-Z0-9+&@#\/%=~_|$?!:,.]*\)|[A-Z0-9+&@#\/%=~_|$])"

    return re.findall(pattern, msg, re.IGNORECASE | re.MULTILINE)


class LinkSummary:
    url: str
    summary: str

    def __init__(self, url: str, summary: str):
        self.url = url
        self.summary = summary


def link_summarizer(links: List[str]) -> Generator[LinkSummary, None, None]:
    if len(links) > 2:
        return

    # Add "http://" if the scheme is missing from the URL
    links = [l if l.startswith("http") else f"http://{l}" for l in links]

    llm = ChatOpenAI(temperature=0)
    text_splitter = CharacterTextSplitter.from_tiktoken_encoder()
    olf = text_splitter._length_function
    text_splitter._length_function = lambda x: olf(x) + olf(prompt_template)

    chain = load_summarize_chain(llm, chain_type="map_reduce")

    loader = PlaywrightURLLoader(urls=links, remove_selectors=["header", "footer", "script", "style", "iframe", "nav"])
    pages = loader.load()

    print(f"Got {len(pages)} pages")
    for page in pages:
        texts = text_splitter.split_text(page.page_content)
        docs = [Document(page_content=t) for t in texts]

        summary = chain.run(docs)
        yield LinkSummary(url=page.metadata['source'], summary=summary)


@message_handler
def handle_message(ctx: Context, msg: Message) -> CommandResult:
    if msg.sent_before(timedelta(minutes=30)):
        return

    chat_jid, err = parse_jid(msg.chat)
    if err is not None:
        logger.warning(f"Failed to parse chat JID: {err}")
        return

    if chat_jid.is_group():
        store: ChatStore = ctx.store
        group = store.get_group(chat_jid)
        if group is None:
            logger.warning(f"Failed to get group {chat_jid}: {err}")
            return

        if not group.managed:
            logger.debug(f"Group {chat_jid} is not managed, but found a link. Ignoring.")
            return

    links = find_links(msg.text)
    if len(links) > 0:
        logger.debug(f"Found {len(links)} links")
        for summary in link_summarizer(links):
            yield msg_cmd(msg.chat, f"Yo, a quick summary for {summary.url}:\n{summary.summary}",
                          reply_to=msg.raw_message_event)
