# How to parse a web page content?
from langchain.document_loaders import WebBaseLoader

# @TODO replace this by https://geers.in/ai
loader = WebBaseLoader("https://lilianweng.github.io/posts/2023-06-23-agent/")
data = loader.load()[0]

print(data.metadata)
print(data.page_content)