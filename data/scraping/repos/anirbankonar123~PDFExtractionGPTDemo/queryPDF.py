from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
from langchain.callbacks import get_openai_callback
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import PyPDFLoader

def process_text(text):
    #Split the text into chunks using Langchain's CharacterTextSplitter
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)

    # Convert the chunks of text into embeddings to add to FAISS Vector Index
    embeddings = OpenAIEmbeddings()
    knowledgeBase = FAISS.from_texts(chunks, embeddings)

    return knowledgeBase

def process_pages(pages):

    embeddings = OpenAIEmbeddings()
    knowledgeBase = FAISS.from_documents(pages, embeddings)

    return knowledgeBase


model = ChatOpenAI(model_name="gpt-3.5-turbo-16k")

loader = PyPDFLoader("/home/anish/Downloads/IPCC_AR6_SYR_SPM.pdf")
documents = loader.load()
pages = loader.load_and_split()

text = ""
for doc in documents:
    text += doc.page_content

knowledgeBase = process_text(text) # OR use process_pages(pages)

print("knowledge base created")

query = "How much was the global temperature change in 2011 as compared to 1850"

docs = knowledgeBase.similarity_search(query)

#print(docs[0])

llm = OpenAI()
qa_chain = load_qa_chain(llm, chain_type='stuff')
chain = RetrievalQA.from_llm(llm, retriever=knowledgeBase.as_retriever())
response=""
response_qa=""
with get_openai_callback() as cost:
    response_qa = qa_chain.run(input_documents=docs, question=query)
    response = chain(query, return_only_outputs=True)
    print(cost)

print(response)
print(response_qa)

