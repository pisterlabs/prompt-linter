import os
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv(), override=True)

def insert_or_fetch_embeddings(index_name, chunks=None):
    import pinecone
    from langchain.vectorstores import Pinecone
    from langchain.embeddings.openai import OpenAIEmbeddings

    embeddings = OpenAIEmbeddings()

    pinecone.init(api_key=os.environ.get('PINECONE_API_KEY'), environment=os.environ.get('PINECONE_ENV'))

    # Load vector store
    if index_name in pinecone.list_indexes():
        print(f'Index {index_name} already exists. Loading embeddings ...')
        vector_store = Pinecone.from_existing_index(index_name, embeddings)
    else:
        print(f'Index {index_name} does not exist. Creating index and embeddings')
        pinecone.create_index(index_name, dimension=1536, metric='cosine')
        vector_store = Pinecone.from_documents(documents=chunks, embedding=embeddings, index_name=index_name)

    return vector_store

def ask_with_memory_and_prompt(vectorstore, question, chat_history=[]):
    from langchain.llms import OpenAI
    from langchain.chains import ConversationalRetrievalChain
    from langchain.chains import LLMChain
    from langchain.chains.conversational_retrieval.prompts import CONDENSE_QUESTION_PROMPT
    from langchain.chains.qa_with_sources import load_qa_with_sources_chain

    llm = OpenAI(temperature=0)
    question_generator = LLMChain(llm=llm, prompt=CONDENSE_QUESTION_PROMPT) # To get the question as a vector
    doc_chain = load_qa_with_sources_chain(llm, chain_type="map_reduce")

    chain = ConversationalRetrievalChain(
        retriever=vectorstore.as_retriever(),
        question_generator=question_generator,
        combine_docs_chain=doc_chain,
        return_source_documents=True
    )
    chat_history = []
    result = chain({"question": question, "chat_history": chat_history})

    return result
