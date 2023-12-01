import os
import openai
import sys

from langchain.chat_models import ChatOpenAI

sys.path.append('../..')

from dotenv import load_dotenv, find_dotenv
from langchain.document_loaders.generic import GenericLoader
from langchain.document_loaders.parsers import OpenAIWhisperParser
from langchain.document_loaders.blob_loaders.youtube_audio import YoutubeAudioLoader
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter

_ = load_dotenv(find_dotenv())  # read local .env file

openai.api_key = os.environ['OPENAI_API_KEY']

# step 1: load the vedio and audio from YouTube
# Andrew Ng: Introduction to large language models
url = "https://www.youtube.com/watch?v=BunESRhYhec"

save_dir = "docs/youtube/"
loader = GenericLoader(
    YoutubeAudioLoader([url], save_dir),
    OpenAIWhisperParser()
)
docs = loader.load()
print('The content of the docs is' + docs[0].page_content)
print("There are " + str(len(docs)) + " documents in the list of docs")

# step 2: split the vedio into chunks
r_text_splitter = RecursiveCharacterTextSplitter(
    separators=["\n\n", "\n", " ", ""],
    chunk_size=1000,
    chunk_overlap=50,
    length_function=len
)

splits = r_text_splitter.split_documents(docs)

print("There are " + str(len(splits)) + " splits in the list of splits")

for i in range(len(splits)):
    print('text of the ' + str(i) + ' chunk is ' + splits[i].page_content[0:1000])


def pretty_print_docs(docs):
    print(f"\n{'-' * 100}\n".join([f"Document {i + 1}:\n\n" + d.page_content for i, d in enumerate(docs)]))


# step 3: embed the chunks

from langchain.embeddings.openai import OpenAIEmbeddings

embedding = OpenAIEmbeddings()

# Vectorstores
from langchain.vectorstores import Chroma

persist_directory = 'docs/chroma/'

vectordb = Chroma.from_documents(
    documents=splits,
    embedding=embedding,
    persist_directory=persist_directory
)

vectordb.persist()
print('There are ', vectordb._collection.count(), ' documents in the vectorstore')

llm = ChatOpenAI(temperature=0)

# combine contextual compression with self-query
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor

compressor = LLMChainExtractor.from_llm(llm)
#  combining compression and self-query
compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=vectordb.as_retriever(search_type="mmr")
)
question = "What are we going to learn from this course?"
compressed_docs = compression_retriever.get_relevant_documents(question)
pretty_print_docs(compressed_docs)

# Q&A with the documents using the retrieval QA with chain type = REFINE
from langchain.chains import RetrievalQA

qa_chain_refine = RetrievalQA.from_chain_type(
    llm,
    retriever=vectordb.as_retriever(),
    return_source_documents=True,
    chain_type="refine"
)
result = qa_chain_refine({"query": question})
if len(result) > 0:
    print('Result from the RetrievalQA, with chain type = REFINE is ' +  result["result"])

# Q&A with the documents using the retrieval QA with chain type = map - reduce

qa_chain_mr = RetrievalQA.from_chain_type(
    llm,
    retriever=vectordb.as_retriever(),
    return_source_documents=True,
    chain_type="map_reduce"
)
result = qa_chain_mr({"query": question})
print('The result from the RetrievalQA with chain_type = map-reduce is ' + result["result"])

# Use a prompt template to generate a question
# Build prompt
from langchain.prompts import PromptTemplate

template = """Use the following pieces of context to answer the question at the end. If you don't know the answer, 
just say that you don't know, don't try to make up an answer. Use three sentences maximum. Keep the answer as concise 
as possible. Always say "thanks for asking!" at the end of the answer. {context} Question: {question} Helpful Answer:"""
QA_CHAIN_PROMPT = PromptTemplate(input_variables=["context", "question"], template=template, )

# Run chain
from langchain.chains import RetrievalQA

question = "What are the pre requisites for this course?"
qa_chain = RetrievalQA. \
    from_chain_type(llm,
                    retriever=vectordb.as_retriever(),
                    return_source_documents=True,
                    chain_type_kwargs={"prompt": QA_CHAIN_PROMPT})

result = qa_chain({"query": question})
print('Result with a prompt template is ' + result["result"])

#  Use memory to have a chatbot feeling
from langchain.memory import ConversationBufferMemory

memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True
)

from langchain.chains import ConversationalRetrievalChain

retriever = vectordb.as_retriever()
qa = ConversationalRetrievalChain.from_llm(
    llm,
    retriever=retriever,
    memory=memory,
    return_source_documents=True
)

question = "Is LangChain a course topic?"
result = qa({"question": question})

print('Answer from the chat bot with memory is' + result['answer'])

question = "why are those prerequisites needed?"
result = qa({"question": question})

print('Second answer taking the previous answer into consideration is ' + result['answer'])