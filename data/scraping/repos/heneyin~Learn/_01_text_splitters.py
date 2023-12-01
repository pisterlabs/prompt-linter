"""
https://python.langchain.com/docs/modules/data_connection/document_transformers/
"""
# 当您想要处理长文本时，有必要将该文本分割成块。
# 这听起来很简单，但这里存在很多潜在的复杂性。
# 理想情况下，您希望将语义相关的文本片段保留在一起。
# “语义相关”的含义可能取决于文本的类型。本笔记本展示了实现此目的的几种方法。

# 在较高层次上，文本分割器的工作原理如下
# 1. 将文本分成小的、具有语义意义的块（通常是句子）。
# 2. 开始将这些小块组合成一个更大的块，直到达到一定的大小（通过某些函数测量）。
# 3. 一旦达到该大小，请将该块设为自己的文本片段，然后开始创建具有一些重叠的新文本块（以保持块之间的上下文）。

# 这意味着您可以沿着两个不同的 axes 自定义文本拆分器：
# 1. How the text is split`
# 2. How the chunk size is measured

"""
默认推荐的文本分割器是 RecursiveCharacterTextSplitter。
"""

# This is a long document we can split up.
with open('../../texts/maodun.txt') as f:
    state_of_the_union = f.read()

from langchain.text_splitter import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(
    # Set a really small chunk size, just to show.
    chunk_size=100,
    chunk_overlap=20,
    length_function=len,
    add_start_index=True,
)

# 创建切分后的文档
texts = text_splitter.create_documents([state_of_the_union])
print(texts[0])
print(texts[1])
print(texts[2])
print(texts[3])