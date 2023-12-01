from embeddings import load_vector_store_docs, OpenAIEmbeddings
from dotenv import load_dotenv
from chatgpt import chat_gpt
from langchain import OpenAI, Wikipedia
from langchain.agents import Tool
from langchain.llms import OpenAI
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain.chains.query_constructor.base import AttributeInfo
from langchain.agents.react.base import DocstoreExplorer
from langchain.vectorstores import Chroma
docstore = DocstoreExplorer(Wikipedia())

llm = OpenAI(temperature=0)
embeddings = OpenAIEmbeddings()

tools = [
    Tool(
        name="Search",
        func=docstore.search,
        description="useful for when you need to ask with search"
    ),
    Tool(name="Lookup",
         func=docstore.lookup,
         description="useful for when you need to ask with lookup"
         )
]

load_dotenv()

metadata_field_info=[
    AttributeInfo(
        name='README.md',
        description='readme file for the hexamerous project',
        type='string or list[string]'
    )
]

document_content_description = 'a readme file from a python project called hexamerous.'

def base_retriever(user_query):
    vectorstore = load_vector_store_docs()
    retriever = SelfQueryRetriever.from_llm(llm, vectorstore, document_content_description, metadata_field_info, verbose=True)
    docs = retriever.get_relevant_documents(user_query)
    print(docs)
    return docs




def data_base_memory_search(user_query):
    docs = base_retriever(user_query)
    prompt = {
        "role": "system",
        "content": '''
        "The user has asked this question:

        {query}

        You have looked up the relevant information from your data store and it is:

        {data}

        Please answer the user's question using the data as relevant context."
        '''.format(query=user_query, data=docs)
    }
    print(prompt)

    result = chat_gpt(prompt)

    print("Memory search result: " + result)

    return result
