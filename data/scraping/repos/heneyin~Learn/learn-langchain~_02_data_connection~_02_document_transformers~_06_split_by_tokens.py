"""
模型会有 token 限制。
按照 token num 分割。
"""

"""
openai 的分割器
"""

# This is a long document we can split up.
with open("../../texts/maodun.txt") as f:
    state_of_the_union = f.read()
from langchain.text_splitter import CharacterTextSplitter

text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=100, chunk_overlap=0
)
texts = text_splitter.split_text(state_of_the_union)

print(texts[0])
print(texts[1])
print(texts[2])

# We can also load a tiktoken splitter directly

from langchain.text_splitter import TokenTextSplitter

text_splitter = TokenTextSplitter(chunk_size=10, chunk_overlap=0)

texts = text_splitter.split_text(state_of_the_union)
print(texts[0])
print(texts[1])
print(texts[2])