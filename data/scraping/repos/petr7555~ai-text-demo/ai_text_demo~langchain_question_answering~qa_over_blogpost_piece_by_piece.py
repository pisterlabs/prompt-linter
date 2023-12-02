import logging

import dotenv
from langchain import PromptTemplate
from langchain.chains import RetrievalQA, ConversationalRetrievalChain
from langchain.chains.question_answering import load_qa_chain
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import WebBaseLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.memory import ConversationSummaryMemory
from langchain.retrievers import SVMRetriever
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain import hub
from langchain.chains import RetrievalQAWithSourcesChain

# Load OPENAI_API_KEY
dotenv.load_dotenv()

# Step 1. Load
loader = WebBaseLoader("https://lilianweng.github.io/posts/2023-06-23-agent/")
data = loader.load()

# Step 2. Split
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
all_splits = text_splitter.split_documents(data)

# Step 3. Store
vectorstore = Chroma.from_documents(documents=all_splits, embedding=OpenAIEmbeddings())

# Step 4. Retrieve
# Using vectorstore
question = "What are the approaches to Task Decomposition?"
docs = vectorstore.similarity_search(question)
print(f"Found {len(docs)} documents with vectorstore similarity search.")

# Using SVM
svm_retriever = SVMRetriever.from_documents(all_splits, OpenAIEmbeddings())
docs_svm = svm_retriever.get_relevant_documents(question)
print(f"Found {len(docs_svm)} documents with SVM.")

# MultiQueryRetriever generates variants of the input question to improve retrieval.
logging.basicConfig()
logging.getLogger('langchain.retrievers.multi_query').setLevel(logging.INFO)

retriever_from_llm = MultiQueryRetriever.from_llm(retriever=vectorstore.as_retriever(),
                                                  llm=ChatOpenAI(temperature=0))
unique_docs = retriever_from_llm.get_relevant_documents(query=question)
print(f"Found {len(unique_docs)} documents with MultiQueryRetriever.")

# Step 5. Generate
llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
qa_chain = RetrievalQA.from_chain_type(llm, retriever=vectorstore.as_retriever())
print("Basic question answering")
print(qa_chain({"query": question}))

# Customize the prompt
template = """Use the following pieces of context to answer the question at the end. 
If you don't know the answer, just say that you don't know, don't try to make up an answer. 
Use three sentences maximum and keep the answer as concise as possible. 
Always say "thanks for asking!" at the end of the answer. 
{context}
Question: {question}
Helpful Answer:"""
QA_CHAIN_PROMPT = PromptTemplate.from_template(template)

qa_chain = RetrievalQA.from_chain_type(
    llm,
    retriever=vectorstore.as_retriever(),
    chain_type_kwargs={"prompt": QA_CHAIN_PROMPT}
)
result = qa_chain({"query": question})
print("Customized prompt")
print(result["result"])

# RAG prompt
# https://smith.langchain.com/hub/rlm/rag-prompt
QA_CHAIN_PROMPT_HUB = hub.pull("rlm/rag-prompt")

qa_chain = RetrievalQA.from_chain_type(
    llm,
    retriever=vectorstore.as_retriever(),
    chain_type_kwargs={"prompt": QA_CHAIN_PROMPT_HUB}
)
result = qa_chain({"query": question})
print("RAG prompt")
print(result["result"])

# Return source documents
qa_chain = RetrievalQA.from_chain_type(llm, retriever=vectorstore.as_retriever(),
                                       return_source_documents=True)
result = qa_chain({"query": question})
print("Return source documents")
print(f"The result is based on {len(result['source_documents'])} source documents.")
print("The first source document is:")
print(result['source_documents'][0])

# Return citations (sources)
qa_chain = RetrievalQAWithSourcesChain.from_chain_type(llm, retriever=vectorstore.as_retriever())

result = qa_chain({"question": question})
print("Return citations (sources)")
print(result)

# Customizing retrieved document processing
chain = load_qa_chain(llm, chain_type="stuff")
print("Pass documents to an LLM prompt using the chain_type 'stuff'")
print(chain({"input_documents": unique_docs, "question": question}, return_only_outputs=True))

# Or pass the chain_type to RetrievalQA
qa_chain = RetrievalQA.from_chain_type(llm, retriever=vectorstore.as_retriever(),
                                       chain_type="stuff")
result = qa_chain({"query": question})
print("Or pass the chain_type to RetrievalQA")
print(result["result"])

# Step 6. Chat
memory = ConversationSummaryMemory(llm=llm, memory_key="chat_history", return_messages=True)
qa = ConversationalRetrievalChain.from_llm(llm, retriever=vectorstore.as_retriever(), memory=memory, verbose=True)
print(qa("How do agents use Task decomposition?"))
print(qa("What are the various ways to implement memory to support it?"))
