import hashlib
import os
import time
from bs4 import BeautifulSoup as Soup
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma

from sky_langchain.loaders.recursive_url_loader import RecursiveUrlLoader
from sky_langchain.models.chatgpt import chatgpt_three_point_five_turbo
from sky_langchain.wrapper import ChatChainWrapper
from bs4 import BeautifulSoup as Soup


def scraper(
    name="marx",
    url="https://www.marxists.org/chinese/marx/capital/",
    domain=None,
):
    args = dict(
        url=url,
        domain=domain or url,
        max_depth=2,
        extractor=lambda x: Soup(x, "html.parser").get_text(),
    )
    cache_dir = os.path.join(".embeddings/", name)
    llm = chatgpt_three_point_five_turbo
    embedding = OpenAIEmbeddings()
    if not os.path.exists(cache_dir):
        loader = RecursiveUrlLoader(**args)
        data = loader.load()

        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
        splits = splitter.split_documents(data)
        vstore = Chroma(
            collection_name="my_collection",
            embedding_function=embedding,
            persist_directory=cache_dir,
        )
        batch_size = 300
        total = len(splits)
        progress = 0
        for split in range(0, len(splits), batch_size):
            vstore.add_documents(splits[split : split + batch_size])
            progress += batch_size
            print(f"{progress}/{total}")
            time.sleep(30)
    else:
        vstore = Chroma(
            collection_name="my_collection",
            embedding_function=embedding,
            persist_directory=cache_dir,
        )

    retriever = vstore.as_retriever(search_type="mmr")
    wrapper = ChatChainWrapper(llm, retriever, chinese=True)
    wrapper.run()


# scraper(name="langchain_docs", url="https://python.langchain.com.cn/docs/", domain="https://python.langchain.com.cn/docs/)
# scraper()
