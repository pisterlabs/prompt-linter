# This is a demo version using a pdf loader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.retrievers import SVMRetriever
from langchain.chat_models import ChatOpenAI
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.chains.question_answering import load_qa_chain
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.document_loaders import DirectoryLoader


# loads a pdf into a document object
loader = DirectoryLoader('/Users/ayushjain/OneSpace/Untitled/pdf/', glob="./*.pdf", loader_cls=PyPDFLoader)

#loader = PyPDFLoader("/Users/ayushjain/Downloads/case 1.pdf")
data = loader.load()

# the other option is the character text splitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500, chunk_overlap=0, length_function=len,)
# the splits are fed into the embedder and then fed into the vector store
all_splits = text_splitter.split_documents(data)

# creates the vector store using the OPENAI Embedding
vectorstore = Chroma.from_documents(
    documents=all_splits, embedding=OpenAIEmbeddings())

# similarity search retrieves the relevant documents
question = "What are the rules for a brief opposing a motion?"

# can do search with a vectorstore or with other methods
# they all implement the common method get_relevant_documents()
docs = vectorstore.similarity_search(question)

# example with SVM !Does not work yet
# svm_retriever = SVMRetriever.from_documents(all_splits,OpenAIEmbeddings())
# docs_svm=svm_retriever.get_relevant_documents(question)
# len(docs_svm)

# MultiQueryRetriever: https://python.langchain.com/docs/modules/data_connection/retrievers/how_to/MultiQueryRetriever

##### FIGURE OUT WHERE THE MULTIQUERY RETRIEVER FITS IN ###
retriever_from_llm = MultiQueryRetriever.from_llm(retriever=vectorstore.as_retriever(),  # the retriever is the vectorStore
                                                  llm=ChatOpenAI(temperature=0))  # it asks the query in numerous ways to gpt to create the most optimal + consistent answer
unique_docs = retriever_from_llm.get_relevant_documents(query=question)
# print("UNIQUE DOCUMENTS: ") 
# print(unique_docs)


# MaxMarginalRelevance: https://www.cs.cmu.edu/~jgc/publication/The_Use_MMR_Diversity_Based_LTMIR_1998.pdf
# Metadata Filters: https://python.langchain.com/docs/use_cases/question_answering/document-context-aware-QA

# each doc is not a document but a split, it has its own page and source metadata that can be parsed
# for doc in unique_docs:
#     print("This is a doc ___________________: ")
#     print(doc)
#     print(doc.metadata)
#     print(doc.metadata['page'])

# prompt templates are also useful

# template = """Use the following pieces of context to answer the question at the end. 
# If you don't know the answer, just say that you don't know, don't try to make up an answer. 
# Use three sentences maximum and keep the answer as concise as possible. 
# Always say "thanks for asking!" at the end of the answer. 
# {context}
# Question: {question}
# Helpful Answer:"""

# QA_CHAIN_PROMPT = PromptTemplate.from_template(template)


# can convert the relevant documents into an answer using an LLM model
# play around with the temperature to change the truth value

# can use a local LLM like GPT4All
llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
# qa_chain = RetrievalQA.from_chain_type(
#     llm,
#     retriever=vectorstore.as_retriever(),
#     chain_type_kwargs={"prompt": QA_CHAIN_PROMPT},
#     return_source_documents=True #allows us to find where the source is
# )
# result = qa_chain({"query": question})
# print(result["result"]) #returns the result of documents 
# print(result["source_documents"]) #returns teh source of the documents 

#can have different types of passing documents to an LLM : https://python.langchain.com/docs/modules/chains/document/

# creates a qa chain that takes in the splits as well as the question and then creates an output using a certiain type of document refining 
# chain = load_qa_chain(llm, chain_type="refine")
# chain({"input_documents": unique_docs, "question": question},return_only_outputs=False)

#the memory for a conversation 
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

retriever = vectorstore.as_retriever() #retriever from llm is much slower so possibly stick to normal retriever for now 
chat = ConversationalRetrievalChain.from_llm(llm, retriever=retriever_from_llm, memory=memory)


result = chat({"question": "What are the possible failure points when submitting a motion?"})
# print(result['answer'])
# print("---------------------------------------------------------")
# print(vectorstore.similarity_search_with_relevance_scores(result["answer"])) # docs: https://api.python.langchain.com/en/latest/vectorstores/langchain.vectorstores.chroma.Chroma.html#langchain.vectorstores.chroma.Chroma.similarity_search_with_relevance_scores

result = chat({"question": "What are some ways to not have those failure points?"})
print(result['answer'])

#do a similarity check on the vectorstore
