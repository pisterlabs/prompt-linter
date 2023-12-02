import os
import openai

openai.api_key  = os.environ['OPENAI_API_KEY']

llm_name = "gpt-3.5-turbo"

###############################
# Simple Question Answering ###
###############################

from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
persist_directory = 'docs/chroma/'
embedding = OpenAIEmbeddings()
vectordb = Chroma(persist_directory=persist_directory, embedding_function=embedding)

# Check that vector DB is working, should see something like 209 vectors
print(vectordb._collection.count())

question = "What are major topics for this class?"
docs = vectordb.similarity_search(question,k=3)
len(docs)

from langchain.chat_models import ChatOpenAI
llm = ChatOpenAI(model_name=llm_name, temperature=0) # Set temperature to 0 when we want exact answers with high fidelity

# Define the retrivl chain
from langchain.chains import RetrievalQA
qa_chain = RetrievalQA.from_chain_type(
    llm,
    retriever=vectordb.as_retriever(),
    verbose=True
)

# Commenting this out so I don't run it each time I run the program, if you are following along comment this back in
# result = qa_chain({"query": question})
# print(result["result"])


#########################################################
### Query the document with a custom prompt framework ###
#########################################################

# The default technique for QA is the "stuff technique" where everything goes into one prompt. Good because it is cheaper.
from langchain.prompts import PromptTemplate

# Build prompt
template = """Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer. Use three sentences maximum. Keep the answer as concise as possible. Always say "thanks for asking!" at the end of the answer. 
{context}
Question: {question}
Helpful Answer:"""
QA_CHAIN_PROMPT = PromptTemplate.from_template(template)

# Run chain
qa_chain = RetrievalQA.from_chain_type(
    llm,
    retriever=vectordb.as_retriever(),
    return_source_documents=True,
    chain_type_kwargs={"prompt": QA_CHAIN_PROMPT}
)

question = "What did they say about matlab?"

result = qa_chain({"query": question})

print(result["result"])

print(result["source_documents"][0])
print("")

##################
### Map Reduce ###
##################

# Since the "stuff" technique of stuffing all the documents into one prompt fails when there are too many documents, we can instead use map reduce.
# In Map Reduce, each of the individual documents is sent to the LLM by itself first to get an origian answer
# Then the LLM compiles these answers to get one final answer

# This is much slower and much more expensive.
# This may fail if context is spread across documents

print("Now using map reduce")
qa_chain_mr = RetrievalQA.from_chain_type(
    llm,
    retriever=vectordb.as_retriever(),
    chain_type="map_reduce",
)
result = qa_chain_mr({"query": question})

print(result["result"])
print("")

# Now instead of plain map reduce we use refine
# Refine passes in each document to the LLM one at a time along with the previous oputput and asks the LLM to consider the additional document and refine the answer as needed, as opposed to regular map reduce which processes individually and combines at the end
print("Use map reduce with chain type refine")
qa_chain_mr = RetrievalQA.from_chain_type(
    llm,
    retriever=vectordb.as_retriever(),
    chain_type="refine",
    return_source_documents=True,
)
result = qa_chain_mr({"query": question})
print(result["result"])