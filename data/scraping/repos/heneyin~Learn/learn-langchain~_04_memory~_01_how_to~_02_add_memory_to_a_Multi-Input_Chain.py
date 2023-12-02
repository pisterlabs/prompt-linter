"""
https://python.langchain.com/docs/modules/memory/how_to/adding_memory_chain_multiple_inputs

向多个输入的 chain 添加 memory

下面的例子，向问答 chain 添加 memory, 这个 chain 使用相关的文档与用户的问题作为输入。
"""

import env
from langchain.chat_models import ChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.embeddings.cohere import CohereEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores.elastic_vector_search import ElasticVectorSearch
from langchain.vectorstores import Chroma
from langchain.docstore.document import Document

# 切分 text
with open("../../texts/maodun.txt") as f:
    state_of_the_union = f.read()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
texts = text_splitter.split_text(state_of_the_union)


# 生成向量数据库
embeddings = OpenAIEmbeddings()
docsearch = Chroma.from_texts(
    texts, embeddings, metadatas=[{"source": i} for i in range(len(texts))]
)

# 找到相似的文档
query = "矛盾的特殊性"
docs = docsearch.similarity_search(query)

from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory


template = """You are a chatbot having a conversation with a human.

Given the following extracted parts of a long document and a question, create a final answer.

{context}

{chat_history}
Human: {human_input}
Chatbot:"""

# 提示词，包括文档内容、对话历史、用户输入
prompt = PromptTemplate(
    input_variables=["chat_history", "human_input", "context"], template=template
)
# 记录历史对话
memory = ConversationBufferMemory(memory_key="chat_history", input_key="human_input")
chain = load_qa_chain(
    ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo-16k-0613"), chain_type="stuff", memory=memory, prompt=prompt
)

query = "矛盾的特殊性"
chain({"input_documents": docs, "human_input": query}, return_only_outputs=True)

print("", chain.memory.buffer)