from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain import PromptTemplate
from langchain.chains.question_answering import load_qa_chain
from langchain.vectorstores import Pinecone
from langchain.retrievers.weaviate_hybrid_search import WeaviateHybridSearchRetriever
from .config import OPENAI_API_KEY

#prompt
def get_prompt():
    template = '''You are a tutor with expertise in the subject matter presented in the context below.
    Your goal is to provide clear, structured, and concise answers to the user's queries. 
    Whenever explaining concepts, follow a step-by-step format.
    Incorporate examples to elucidate your points. 
    If you can't address a query based on the provided context, reply with "I don't know".

    - Ensure the response is clear, structured, and concise.
    - Use headers and sub-headers to demarcate different sections of your answer. This is crucial.
    - Answer in a way that faciliates learning
    - If a question cannot be addressed based on the provided context, reply with "I don't know".

    {context}

    Question: {question}

    Answer this question in a markdown format with the above instructions 
    The biggest header size should ONLY be ###
    '''


    PROMPT = PromptTemplate(
        template=template, input_variables=["context", "question"]
    )
    return PROMPT

def run_llm(query: str, docs: object):
    #llm = OpenAI(temperature=0.5, openai_api_key=OPENAI_API_KEY)
    chat = ChatOpenAI(temperature=0.5, openai_api_key=OPENAI_API_KEY, verbose=True)
    PROMPT = get_prompt()
    chain = load_qa_chain(chat, chain_type="stuff", prompt=PROMPT)
    result = chain.run({"input_documents": docs, "question": query})
    print(result)
    return result

#helper functions to choose which vector db
def answer(query: str, docsearch: Pinecone) -> str:
    docs = docsearch.similarity_search(query)
    print(f'k: {len(docs)}')
    for doc in docs:
        print(doc)
        print('_______________________')
    return run_llm(query, docs)

def answer_hybrid(query: str, retriever: WeaviateHybridSearchRetriever) -> str:
    docs = retriever.get_relevant_documents(query)
    for doc in docs:
        print(doc)
        print('_______________________')
    return run_llm(query, docs)
    

