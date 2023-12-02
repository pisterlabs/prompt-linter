# Take the documents from the previous lesson and take the original question
# passing both to LLM and ask it to answer the question

import os, sys
from dotenv import load_dotenv

load_dotenv()

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


from langchain.vectorstores import Chroma
from llms.llm import azure_openai_embeddings, azure_chat_openai_llm

persist_directory = "data/chroma/"
embedding = azure_openai_embeddings()
vectordb = Chroma(persist_directory=persist_directory, embedding_function=embedding)

question = "What are major topics for this class?"
# docs = vectordb.similarity_search(question, k=3)

# print(len(docs))


def retrievalQA():
    from langchain.chains import RetrievalQA

    llm = azure_chat_openai_llm()

    qa_chain = RetrievalQA.from_chain_type(llm, retriever=vectordb.as_retriever())
    result = qa_chain({"query": question})

    print(result["result"])


def retrievalQA_with_prompt():
    from langchain.prompts import PromptTemplate
    from langchain.chains import RetrievalQA

    llm = azure_chat_openai_llm()

    # Build prompt
    template = """Use the following pieces of context to answer the question at the end. 
                  If you don't know the answer, just say that you don't know, don't try to make up an answer. 
                  Use three sentences maximum. 
                  Keep the answer as concise as possible. 
                  Always say "thanks for asking!" at the end of the answer. 
    {context}
    Question: {question}
    Helpful Answer:"""

    QA_CHAIN_PROMPT = PromptTemplate.from_template(template)
    question = "Is probability a class topic?"

    # Run chain
    # SCENARIO 1 Stuff chain
    # limitation for the default chain type(stuff), if documents are many , we can reach the limit of tokens
    # because this kind of chain type sends all the documents to the model at once
    # qa_chain = RetrievalQA.from_chain_type(
    #     llm,
    #     retriever=vectordb.as_retriever(),
    #     return_source_documents=True,
    #     chain_type_kwargs={"prompt": QA_CHAIN_PROMPT},
    # )

    # SCENARIO 2 map_reduce chain
    # map_reduce chain type sends one document at a time to the model for getting the original answer
    # then those answers are composed into a final answer with a final call to the model
    # This teqnique involves many calls to the model but it can operate over arbitraily many documents
    # 2 limitations :
    # - slow
    # - no clear answer to the question, because it's answering on each document individually

    qa_chain = RetrievalQA.from_chain_type(
        llm,
        retriever=vectordb.as_retriever(),
        return_source_documents=True,
        chain_type="map_reduce",
    )

    # SCENARIO 3 refine chain
    # newest and better because use a different approach to the problem
    # see doc for more details
    qa_chain = RetrievalQA.from_chain_type(
        llm,
        retriever=vectordb.as_retriever(),
        return_source_documents=True,
        chain_type="refine",
    )

    result = qa_chain({"query": question})
    print(len(result["source_documents"]))
    print(result["result"])


retrievalQA_with_prompt()
