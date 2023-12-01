""" Import files into Convex using Langchain document loaders

Setup:
!pip install "playwright"
!pip install "unstructured"
!pip install "convex"
!pip install "python-dotenv"
!pip install tiktoken

!playwright install
"""

import os, sys
from dotenv import load_dotenv
from convex import ConvexClient
from langchain.document_loaders import PlaywrightURLLoader
from langchain.text_splitter import CharacterTextSplitter


urls = sys.argv[1:]
loader = PlaywrightURLLoader(urls=urls, remove_selectors=["header", "footer"])
data = loader.load()
text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=100, chunk_overlap=0
)
texts = text_splitter.split_text(data[0].page_content)


load_dotenv(".env.local")
load_dotenv()

backend = os.getenv("CONVEX_URL")
if not backend:
    raise KeyError("Missing CONVEX_URL")

client = ConvexClient(backend)
print(
    client.action(
        "sources:add",
        dict(
            name=data[0].metadata["source"],
            chunks=list(
                map(
                    lambda chunk: dict(
                        text=chunk,
                        # TODO: add real line numbers
                        lines={"from": 0, "to": 0},
                    ),
                    texts,
                )
            ),
        ),
    )
)
