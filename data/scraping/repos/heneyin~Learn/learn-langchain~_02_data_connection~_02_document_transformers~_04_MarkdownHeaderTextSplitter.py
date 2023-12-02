"""
https://python.langchain.com/docs/modules/data_connection/document_transformers/text_splitters/markdown_header_metadata

许多聊天或问答应用程序都涉及在嵌入和矢量存储之前对输入文档进行分块。

当嵌入完整的段落或文档时，嵌入过程会考虑整体上下文以及文本中句子和短语之间的关系。 这可以产生更全面的矢量表示，捕获文本的更广泛的含义和主题。

如前所述，分块通常旨在将具有共同上下文的文本放在一起。
考虑到这一点，我们可能希望特别尊重文档本身的结构。
例如，Markdown 文件是按标题组织的。

在特定标头组中创建块是一个直观的想法。
为了解决这个挑战，我们可以使用 MarkdownHeaderTextSplitter。

这将按一组指定的 header 拆分 Markdown 文件。

例如，如果我们想分割这个 markdown：

```
md = '# Foo\n\n ## Bar\n\nHi this is Jim  \nHi this is Joe\n\n ## Baz\n\n Hi this is Molly'
```

We can specify the headers to split on:


```
[("#", "Header 1"),("##", "Header 2")]
```

"""

from langchain.text_splitter import MarkdownHeaderTextSplitter

markdown_document = "# Foo\n\n    ## Bar\n\nHi this is Jim\n\nHi this is Joe\n\n ### Boo \n\n Hi this is Lance \n\n ## Baz\n\n Hi this is Molly"
markdown_document = "# Foo\n\n    ## Bar\n\nHi this is Jim\n\nHi this is Joe\n\n ### Boo \n\n Hi this is Lance \n\n ## Baz\n\n Hi this is Molly"

headers_to_split_on = [
    ("#", "Header 1"),
    ("##", "Header 2"),
    ("###", "Header 3"),
]

markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
md_header_splits = markdown_splitter.split_text(markdown_document)

for i in md_header_splits:
    print(i)

"""
Within each markdown group we can then apply any text splitter we want.
按照 header 分组后再进行切分。
"""
markdown_document = "# Intro \n\n    ## History \n\n Markdown[9] is a lightweight markup language for creating formatted text using a plain-text editor. John Gruber created Markdown in 2004 as a markup language that is appealing to human readers in its source code form.[9] \n\n Markdown is widely used in blogging, instant messaging, online forums, collaborative software, documentation pages, and readme files. \n\n ## Rise and divergence \n\n As Markdown popularity grew rapidly, many Markdown implementations appeared, driven mostly by the need for \n\n additional features such as tables, footnotes, definition lists,[note 1] and Markdown inside HTML blocks. \n\n #### Standardization \n\n From 2012, a group of people, including Jeff Atwood and John MacFarlane, launched what Atwood characterised as a standardisation effort. \n\n ## Implementations \n\n Implementations of Markdown are available for over a dozen programming languages."

headers_to_split_on = [
    ("#", "Header 1"),
    ("##", "Header 2"),
]

# MD splits
markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
md_header_splits = markdown_splitter.split_text(markdown_document)

# Char-level splits
from langchain.text_splitter import RecursiveCharacterTextSplitter
chunk_size = 250
chunk_overlap = 30
text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

# Split
splits = text_splitter.split_documents(md_header_splits)

for i in splits:
    print(i)




