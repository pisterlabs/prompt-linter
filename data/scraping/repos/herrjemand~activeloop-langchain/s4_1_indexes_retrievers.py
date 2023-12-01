from dotenv import load_dotenv
load_dotenv(dotenv_path='.env')
import os

from langchain.document_loaders import TextLoader

# text to write to a local file
# taken from https://www.theverge.com/2023/3/14/23639313/google-ai-language-model-palm-api-challenge-openai
text = """Google opens up its AI language model PaLM to challenge OpenAI and GPT-3
Google is offering developers access to one of its most advanced AI language models: PaLM.
The search giant is launching an API for PaLM alongside a number of AI enterprise tools
it says will help businesses “generate text, images, code, videos, audio, and more from
simple natural language prompts.”

PaLM is a large language model, or LLM, similar to the GPT series created by OpenAI or
Meta’s LLaMA family of models. Google first announced PaLM in April 2022. Like other LLMs,
PaLM is a flexible system that can potentially carry out all sorts of text generation and
editing tasks. You could train PaLM to be a conversational chatbot like ChatGPT, for
example, or you could use it for tasks like summarizing text or even writing code.
(It’s similar to features Google also announced today for its Workspace apps like Google
Docs and Gmail.)
"""

with open("example.txt", "w") as f:
    f.write(text)

loader = TextLoader("example.txt")
docs_from_file = loader.load()

from langchain.text_splitter import CharacterTextSplitter

text_splitter = CharacterTextSplitter(chunk_size=200, chunk_overlap=20)

docs = text_splitter.split_documents(docs_from_file)

from langchain.embeddings import OpenAIEmbeddings

embeddings = OpenAIEmbeddings()

from langchain.vectorstores import DeepLake

activeloop_dataset_name = "langchain_course_from_zero_to_hero"
dataset_path = f"hub://{os.environ.get('ACTIVELOOP_ORGID')}/{activeloop_dataset_name}"
vecdb = DeepLake(dataset_path=dataset_path, embedding=embeddings)

vecdb.add_documents(docs)

retriever = vecdb.as_retriever()

from langchain.chains import RetrievalQA
from langchain.llms import OpenAI

llm = OpenAI(model="text-davinci-003", temperature=0.0)

retrieval_qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
)

query = "How Google plans to challenge OpenAI?"
response = retrieval_qa.run(query)
print(response)


from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor

llm = OpenAI(model="text-davinci-003", temperature=0.0)

compressor = LLMChainExtractor.from_llm(llm)
compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=retriever,
)

retrieved_docs = compression_retriever.get_relevant_documents("How google plans to challenge OpenAI?")
print(retrieved_docs[0].page_content)