from langchain.chat_models import BedrockChat
from langchain.embeddings import BedrockEmbeddings
from langchain.prompts import PromptTemplate
from langchain.schema import HumanMessage
from langchain.vectorstores import Chroma
import sys
__import__('pysqlite3')
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

embeddings = BedrockEmbeddings(
    model_id="amazon.titan-embed-text-v1"
)

database = Chroma(
    persist_directory="./.data",
    embedding_function=embeddings
)

query = "飛行車の最高速度は？"

documents = database.similarity_search(query)

documents_string = ""

for document in documents:
    documents_string += f"""
----------------------------
{document.page_content}
"""

prompt = PromptTemplate(
    template="""文章を元に質問に答えてください。

文章：
{document}

質問： {query}
""",
    input_variables=["document", "query"]
)

chat = BedrockChat(
    model_id="anthropic.claude-v2"
)

result = chat([
    HumanMessage(content=prompt.format(document=documents_string, query=query))
])

print(result.content)
