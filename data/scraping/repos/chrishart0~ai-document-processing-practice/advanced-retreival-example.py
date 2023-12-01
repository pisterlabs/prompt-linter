# Original Source: https://learn.deeplearning.ai/langchain-chat-with-your-data/lesson/5/retrieval
import os
import openai
import shutil

openai.api_key  = os.environ['OPENAI_API_KEY']

from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
persist_directory = 'docs/chroma/'

# Try to remove the tree; if it fails, throw an error using try...except.
# if os.path.exists(persist_directory) and os.path.isdir(persist_directory):
#     print("Deleting old choma data store")
#     shutil.rmtree(persist_directory)

# Prepare OpenAI and Chroma, chroma will use the previously created vector DB
embedding = OpenAIEmbeddings()
vectordb = Chroma(
    persist_directory=persist_directory,
    embedding_function=embedding
)

print(vectordb._collection.count())


#######################################################
## Small mushroom example of advanced vector search ###
#######################################################

texts = [
    """The Amanita phalloides has a large and imposing epigeous (aboveground) fruiting body (basidiocarp).""",
    """A mushroom with a large fruiting body is the Amanita phalloides. Some varieties are all-white.""",
    """A. phalloides, a.k.a Death Cap, is one of the most poisonous of all known mushrooms.""",
]

# Note: commented out to work on next section, if you are following ilong comment this back in
# smalldb = Chroma.from_texts(texts, embedding=embedding)

question = "Tell me about all-white mushrooms with large fruiting bodies"

# Note: commented out to work on next section, if you are following ilong comment this back in
# Query the vector store
# print("First simple vector store query for 2 documents")
# print(smalldb.similarity_search(question, k=2))
# print("")

# print("Use MMR (Maximum Marginal Relivence) to fetch 3 document and return for 2 documents")
# print(smalldb.max_marginal_relevance_search(question,k=2, fetch_k=3))
# print("")

print("-----------------------")
###########################################################################################
### More advanced search of machine learning lectures done in vectors-and-embeddings.py ###
###########################################################################################

question = "what did they say about matlab?"

# Query the vector DB with a basic similarity search
docs_ss = vectordb.similarity_search(question,k=3)

print("Query the ML lecutres with similarity search")
print(docs_ss[0].page_content[:100])
print(docs_ss[1].page_content[:100])
print("")

# Now use max margianl reliveance for the same query
print("Query the ML lecutres with MMR to get more diverse results")
docs_mmr = vectordb.max_marginal_relevance_search(question,k=2, fetch_k=3)
print(docs_mmr[0].page_content[:100])
print(docs_mmr[1].page_content[:100])
print("")
print("-----------------------")
##### Accessing filtered data: example from a specific lecture

question = "what did they say about regression in the third lecture?"

# Specify the source of the data to only be from the 3rd lecture by hand
docs = vectordb.similarity_search(
    question,
    k=3,
)
print("Ask what is said about regression in the 3rd lecture")
for d in docs:
    print(d.metadata)


# Specify the source of the data to only be from the 3rd lecture by hand
docs = vectordb.similarity_search(
    question,
    k=3,
    filter={"source":"data/MachineLearning-Lecture03.pdf"}
)
print("Ask what is said about regression in the 3rd lections but Query only for results in lecture 3")
for d in docs:
    print(d.metadata)

print("")
# Notice how in the first query of about lecture 3 we actually get a repsonse from lecture 2. By using the metadata filter we are able to narrow this down.
print("Notice how in the first query of about lecture 3 we actually get a repsonse from lecture 2. By using the metadata filter we are able to narrow this down.")
print("-----------------------")

#############################################
## LLM infer metadata for query filtering ###
#############################################

# We can use SelfQueryRetriever, which uses an LLM to extract:
#   The query string to use for vector search
#   A metadata filter to pass in as well, determained by the LLM

from langchain.llms import OpenAI
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain.chains.query_constructor.base import AttributeInfo

metadata_field_info = [
    AttributeInfo(
        name="source",
        description="The lecture the chunk is from, should be one of `data/MachineLearning-Lecture01.pdf`, `data/MachineLearning-Lecture02.pdf`, or `data/MachineLearning-Lecture03.pdf`",
        type="string",
    ),
    AttributeInfo(
        name="page",
        description="The page from the lecture",
        type="integer",
    ),
]

document_content_description = "Lecture notes"
llm = OpenAI(temperature=0)

# https://api.python.langchain.com/en/latest/retrievers/langchain.retrievers.self_query.base.SelfQueryRetriever.html
# https://js.langchain.com/docs/modules/data_connection/retrievers/how_to/self_query/#:~:text=A%20self%2Dquerying%20retriever%20is,query%20to%20it's%20underlying%20VectorStore.
retriever = SelfQueryRetriever.from_llm(
    llm,
    vectordb,
    document_content_description,
    metadata_field_info,
    verbose=True
)

question = "what did they say about regression in the third lecture?"

docs = retriever.get_relevant_documents(question)

print("Ask about regression in thrid lection with LangChain SelfQueryRetriever")
for d in docs:
    # print(d)
    print(d.metadata)

print("")
print("---------------------")

#################################################################################
### Compression: Summerizing and combining data queired from the vector store ###
#################################################################################
# We may not want to return the full document to the user, in this case we can use compression. We will ask the LLM to grab the info relvent to the user's question and summerixe it.

from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor

def pretty_print_docs(docs):
    print(f"\n{'-' * 100}\n".join([f"Document {i+1}:\n\n" + d.page_content for i, d in enumerate(docs)]))


# Wrap our vectorstore
llm = OpenAI(temperature=0)
compressor = LLMChainExtractor.from_llm(llm)

compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=vectordb.as_retriever()
)

question = "what did they say about matlab?"
compressed_docs = compression_retriever.get_relevant_documents(question)
pretty_print_docs(compressed_docs)
print("")
print("---------------------")

############################
### Combining techniques ###
############################

# Using MMR will filter out duplicate results from our documents returned
print("Now use contectual compression retriever with mmr")
compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=vectordb.as_retriever(search_type = "mmr")
)

question = "what did they say about matlab?"
compressed_docs = compression_retriever.get_relevant_documents(question)
pretty_print_docs(compressed_docs)