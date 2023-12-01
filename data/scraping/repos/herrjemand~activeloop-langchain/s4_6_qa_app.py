from dotenv import load_dotenv
load_dotenv(dotenv_path='.env')
import os

from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import DeepLake
from langchain.text_splitter import CharacterTextSplitter
from langchain import OpenAI, PromptTemplate
from langchain.document_loaders import SeleniumURLLoader

urls = ['https://beebom.com/what-is-nft-explained/',
        'https://beebom.com/how-delete-spotify-account/',
        'https://beebom.com/how-download-gif-twitter/',
        'https://beebom.com/how-use-chatgpt-linux-terminal/',
        'https://beebom.com/how-delete-spotify-account/',
        'https://beebom.com/how-save-instagram-story-with-music/',
        'https://beebom.com/how-install-pip-windows/',
        'https://beebom.com/how-check-disk-usage-linux/']

# loader = SeleniumURLLoader(urls=urls)
# raw_docs = loader.load()

# text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
# docs = text_splitter.split_documents(raw_docs)

embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")


activeloop_dataset_name = "langchain_course_5_qa"
dataset_path = f"hub://{os.environ.get('ACTIVELOOP_ORGID')}/{activeloop_dataset_name}"
vecdb = DeepLake(dataset_path=dataset_path, embedding=embeddings)

# vecdb.add_documents(docs)

# query = "How to check disk usage in linux?"

# docs = vecdb.similarity_search(query)
# print(docs[0].page_content)

template = """You are an exceptional customer support chatbot that gently answer questions.

You know the following context information.

{chunks_formatted}

Answer to the following question from a customer. Use only information from the previous context information. Do not invent stuff.

Question: {query}

Answer:"""

prompt_template = PromptTemplate(
    input_variables=["chunks_formatted", "query"],
    template=template,
)

query = "How to check disk usage in linux?"

docs = vecdb.similarity_search(query)
retrieved_chunks = [doc.page_content for doc in docs]

chunks_formatted = "\n\n".join(retrieved_chunks)
prompt_formatted = prompt_template.format(chunks_formatted=chunks_formatted, query=query)

llm = OpenAI(model="text-davinci-003", temperature=0)
answer = llm(prompt_formatted)
print(answer)