"""
https://python.langchain.com/docs/modules/data_connection/document_transformers/text_splitters/recursive_text_splitter

对于一般文本，推荐使用此文本分割器。

It is parameterized by a list of characters.

它尝试按顺序分割它们，直到块足够小。

默认列表为 ["\n\n", "\n", " ", ""]。 这样做的效果是尝试将所有段落（然后是句子，然后是单词）尽可能长地放在一起，因为这些通常看起来是语义相关性最强的文本片段。

1. How the text is split: by list of characters
2. How the chunk size is measured: by number of characters

"""

with open('../../texts/maodun.txt') as f:
    state_of_the_union = f.read()

from langchain.text_splitter import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(
    # Set a really small chunk size, just to show.
    chunk_size=100,
    chunk_overlap=20,
    length_function=len,
)

# 切分为 document
texts = text_splitter.create_documents([state_of_the_union])
print("------------create_documents---------------")
print(texts[0])
print(texts[1])
print(texts[2])
print(texts[3])
print(texts[4])

# 切分为普通文本
result = text_splitter.split_text(state_of_the_union)[:5]
print("------------split_text---------------")
print("result", result)
