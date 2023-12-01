
with open('../quest2.txt', encoding='utf-8') as f:
    state_of_the_union = f.read()

from langchain.text_splitter import TokenTextSplitter

text_splitter = TokenTextSplitter(chunk_size=10, chunk_overlap=0)

texts = text_splitter.split_text(state_of_the_union)
print(texts[0])