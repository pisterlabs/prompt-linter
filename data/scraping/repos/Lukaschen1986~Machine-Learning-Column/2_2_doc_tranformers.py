# -*- coding: utf-8 -*-
"""
https://python.langchain.com/docs/modules/data_connection/document_transformers/
https://www.langchain.com.cn/modules/indexes/text_splitters

Once you've loaded documents, you'll often want to transform them to better suit your application. 
The simplest example is you may want to split a long document into smaller chunks that can fit into your 
model's context window. LangChain has a number of built-in document transformers that make it easy to split, 
combine, filter, and otherwise manipulate documents.
"""
import os
import torch as th
from transformers import GPT2TokenizerFast
from langchain.text_splitter import (RecursiveCharacterTextSplitter, HTMLHeaderTextSplitter,
                                     CharacterTextSplitter, Language, TokenTextSplitter,
                                     SpacyTextSplitter, SentenceTransformersTokenTextSplitter,
                                     PythonCodeTextSplitter)


print(th.cuda.get_device_name())  # NVIDIA GeForce GTX 1080 Ti
device = th.device("cuda" if th.cuda.is_available() else "cpu")

# ----------------------------------------------------------------------------------------------------------------
# 路径
path_project = "C:/my_project/MyGit/Machine-Learning-Column/LangChain"
path_data = os.path.join(os.path.dirname(path_project), "data")
path_model = os.path.join(os.path.dirname(path_project), "model")

# ----------------------------------------------------------------------------------------------------------------
# Split by character
with open('../../../state_of_the_union.txt') as f:
    state_of_the_union = f.read()

text_splitter = CharacterTextSplitter(
    separator = "\n\n",
    chunk_size = 1000,
    chunk_overlap = 200,
    length_function = len,
    is_separator_regex = False,
)
texts = text_splitter.create_documents([state_of_the_union])
print(texts[0])

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 100,
    chunk_overlap = 20,
    length_function = len,
    add_start_index = True,
)
texts = text_splitter.create_documents([state_of_the_union])
print(texts[0])
print(texts[1])

# ----------------------------------------------------------------------------------------------------------------
# With an HTML string
html_string = """
<!DOCTYPE html>
<html>
<body>
    <div>
        <h1>Foo</h1>
        <p>Some intro text about Foo.</p>
        <div>
            <h2>Bar main section</h2>
            <p>Some intro text about Bar.</p>
            <h3>Bar subsection 1</h3>
            <p>Some text about the first subtopic of Bar.</p>
            <h3>Bar subsection 2</h3>
            <p>Some text about the second subtopic of Bar.</p>
        </div>
        <div>
            <h2>Baz</h2>
            <p>Some text about Baz</p>
        </div>
        <br>
        <p>Some concluding text about Foo</p>
    </div>
</body>
</html>
"""

headers_to_split_on = [
    ("h1", "Header 1"),
    ("h2", "Header 2"),
    ("h3", "Header 3"),
]

html_splitter = HTMLHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
html_header_splits = html_splitter.split_text(html_string)
html_header_splits

# ----------------------------------------------------------------------------------------------------------------
# Pipelined to another splitter, with html loaded from a web URL
url = "https://plato.stanford.edu/entries/goedel/"

headers_to_split_on = [
    ("h1", "Header 1"),
    ("h2", "Header 2"),
    ("h3", "Header 3"),
    ("h4", "Header 4"),
]

html_splitter = HTMLHeaderTextSplitter(headers_to_split_on=headers_to_split_on)

# for local file use html_splitter.split_text_from_file(<path_to_file>)
html_header_splits = html_splitter.split_text_from_url(url)

chunk_size = 500
chunk_overlap = 30
text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

# Split
splits = text_splitter.split_documents(html_header_splits)
splits[80:85]

# ----------------------------------------------------------------------------------------------------------------
# Limitations
url = "https://www.cnn.com/2023/09/25/weather/el-nino-winter-us-climate/index.html"

headers_to_split_on = [
    ("h1", "Header 1"),
    ("h2", "Header 2"),
]

html_splitter = HTMLHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
html_header_splits = html_splitter.split_text_from_url(url)
print(html_header_splits[1].page_content[:500])

# ----------------------------------------------------------------------------------------------------------------
# Split code
[e.value for e in Language]  # Full list of support languages

# You can also see the separators used for a given language
RecursiveCharacterTextSplitter.get_separators_for_language(Language.PYTHON)

PYTHON_CODE = """
def hello_world():
    print("Hello, World!")

# Call the function
hello_world()
"""
# python_splitter = RecursiveCharacterTextSplitter.from_language(
#     language=Language.PYTHON, chunk_size=50, chunk_overlap=0
# )
# python_docs = python_splitter.create_documents([PYTHON_CODE])

python_splitter = PythonCodeTextSplitter(chunk_size=30, chunk_overlap=0)
python_docs = python_splitter.create_documents([PYTHON_CODE])
 

# ----------------------------------------------------------------------------------------------------------------
# Split by tokens
'''
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple tiktoken spacy
'''
# This is a long document we can split up.
with open("../../../state_of_the_union.txt") as f:
    state_of_the_union = f.read()

# text_splitter = CharacterTextSplitter.from_tiktoken_encoder(chunk_size=100, chunk_overlap=0)
# texts = text_splitter.split_text(state_of_the_union)
# print(texts[0])

# text_splitter = TokenTextSplitter(chunk_size=10, chunk_overlap=0)
# texts = text_splitter.split_text(state_of_the_union)
# print(texts[0])

# text_splitter = SpacyTextSplitter(chunk_size=1000)
# texts = text_splitter.split_text(state_of_the_union)
# print(texts[0])

# splitter = SentenceTransformersTokenTextSplitter(chunk_overlap=0)
# text = "Lorem "
# count_start_and_stop_tokens = 2
# text_token_count = splitter.count_tokens(text=text) - count_start_and_stop_tokens
# print(text_token_count)

# Hugging Face tokenizer
tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
text_splitter = CharacterTextSplitter.from_huggingface_tokenizer(tokenizer, chunk_size=100, chunk_overlap=0)
texts = text_splitter.split_text(state_of_the_union)
print(texts[0])

# ----------------------------------------------------------------------------------------------------------------
# Lost in the middle: The problem with long contexts
# https://python.langchain.com/docs/modules/data_connection/document_transformers/post_retrieval/long_context_reorder



