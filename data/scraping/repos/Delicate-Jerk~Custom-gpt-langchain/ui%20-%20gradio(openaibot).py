#ui only for the ai bot (nothing else)

import gradio as gr
import os
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA
from langchain.document_loaders import TextLoader
from langchain.document_loaders import DirectoryLoader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# Set up Langchain components (same as in your script)
os.environ["OPENAI_API_KEY"] = "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"
loader = DirectoryLoader(
    '/Users/user1/Downloads/Antier-Sol/5ire/content/DB', glob="./*.txt", loader_cls=TextLoader)
documents = loader.load()
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=200)
texts = text_splitter.split_documents(documents)
embedding = OpenAIEmbeddings()
persist_directory = 'db'
vectordb = Chroma.from_documents(
    documents=texts, embedding=embedding, persist_directory=persist_directory)
vectordb.persist()
vectordb = None
vectordb = Chroma(persist_directory=persist_directory,
                  embedding_function=embedding)
retriever = vectordb.as_retriever()
retriever = vectordb.as_retriever(search_kwargs={"k": 2})
qa_chain = RetrievalQA.from_chain_type(llm=OpenAI(
), chain_type="stuff", retriever=retriever, return_source_documents=True)

# Helper functions


def calculate_similarity(query, response):
    vectorizer = TfidfVectorizer()
    tfidf_query = vectorizer.fit_transform([query])
    tfidf_response = vectorizer.transform([response])
    similarity = cosine_similarity(tfidf_query, tfidf_response)
    return similarity[0][0]


def process_llm_response(query, llm_response):
    return llm_response['result']
    # You can also return similarity if needed

# Define the Gradio interface


def qa_bot(query):
    engineer_prompt = " "
    full_query = " " + query
    llm_response = qa_chain(full_query)
    return process_llm_response(query, llm_response)


# Launch the Gradio interface on your local network
iface = gr.Interface(fn=qa_bot, inputs="text",
                     outputs="text", title="5ire Assistant xD")
iface.launch(share=True)  # Setting share=True enables external access
