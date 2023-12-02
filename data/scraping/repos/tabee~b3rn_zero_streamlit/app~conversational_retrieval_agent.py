import langchain
from langchain.agents.agent_toolkits import (
    create_conversational_retrieval_agent, create_retriever_tool)
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.schema.messages import SystemMessage
from langchain.vectorstores import FAISS
from langchain.cache import SQLiteCache
from langchain.callbacks import get_openai_callback

SYS_PATH_LOCAL = '/workspaces/b3rn_zero_streamlit'
SYS_PATH_STREAMLIT = '/app/b3rn_zero_streamlit/'
SYS_PATH = SYS_PATH_STREAMLIT
langchain.llm_cache = SQLiteCache(database_path=f"{SYS_PATH}/data/langchain_cache.db")

def ask_agent__eak(query, openai_api_key, sys_path, model='gpt-4'):
    '''Display the answer to a question.'''
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)

    new_db = FAISS.load_local(
        f'{sys_path}/data/vectorstores/eak_admin_ch_defaultdocs_faiss_index_4096',
        embeddings)

    retriever = new_db.as_retriever()

    tool = create_retriever_tool(
        retriever,
        "content_of_eak_website",
        """
        This tool is designed for an LLM that interacts with 
        the content of the EAK website to retrieve documents. 
        The EAK acts as a compensation fund for various federal entities. 
        Its main responsibility is overseeing the implementation of 
        the 1st pillar (AHV/IV) and the family compensation fund. 
        The tool offers services related to:
            - Insurance
            - Contributions
            - Employer regulations
            - Pensions
        Furthermore, it provides insights into family allowances and 
        facilitates electronic data exchange with the EAK via connect.eak.
        """
    )
    tools = [tool]

    system_message = SystemMessage(
        content="""
        You are an expert for the eak_admin_website and:
        - Always answer questions citing the source.
        - The source is the URL you receive as a response from the eak_admin_website tool.
        - If you don't know an answer, state: "No source available, thus no answer possible".
        - Never invent URLs. Only use URLs from eak_admin_website.
        - Always respond in German.
        """
    )

    llm = ChatOpenAI(openai_api_key=openai_api_key,
                     model=model,
                     temperature=0,
                     n=10,
                     verbose=True)

    agent_executor = create_conversational_retrieval_agent(
        llm, 
        tools, 
        verbose=False, 
        system_message=system_message,
        max_token_limit=3000) # heikel
 
    print(f"\nFrage: {query}")
    with get_openai_callback() as callback:
        answer = agent_executor({"input": query})
        print(f"\nAntwort: {answer['output']}\n\n")
        print(f"Total Tokens: {callback.total_tokens}")
        print(f"Prompt Tokens: {callback.prompt_tokens}")
        print(f"Completion Tokens: {callback.completion_tokens}")
        print(f"Total Cost (USD): ${callback.total_cost}")
    return answer


def ask_agent__chch(query, openai_api_key, sys_path, model='gpt-4'):
    '''Display the answer to a question.'''
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)

    # new_db1 = FAISS.load_local(
    #     f'{sys_path}/data/vectorstores/eak_admin_ch_defaultdocs_faiss_index_4096',
    #     embeddings)
    # new_db2 = FAISS.load_local(
    #      f'{sys_path}/data/vectorstores/eak_admin_ch_defaultdocs_faiss_index_512',
    #      embeddings)
    
    new_db3 = FAISS.load_local(
        f'{sys_path}/data/vectorstores/ch_ch_texts_faiss_index_4096', 
        embeddings)

    # new_db1.merge_from(new_db2)
    # new_db1.merge_from(new_db3)
    new_db = new_db3

    retriever = new_db.as_retriever()

    tool = create_retriever_tool(
        retriever,
        "content_of_chch_website",
        """
        This tool is designed for an LLM that interacts with 
        the content of the ch.ch website to retrieve documents. 
        The chch acts as a information hub for various federal entities. 
        A service of the Confederation, cantons and communes.
        The tool offers services related to:
        "Easy answers about life in Switzerland"
        The ch.ch portal is an information platform provided by 
        the Swiss authorities. In just a few clicks, you will find 
        straightforward answers in five languages to questions 
        that many of you ask the authorities.
        """
    )
    tools = [tool]

    system_message = SystemMessage(
        content="""
        You are an expert on the chch_website and:
        - Always answer questions by citing the source.
        - The source is the URL you receive as an answer from the content_of_chch_website tool.
        - If you do not know an answer, indicate "No source available, therefore no answer possible".
        - Never make up URLs. Only use URLs from the content_of_chch_website.
        - Always answer in German.
        """
    )

    llm = ChatOpenAI(openai_api_key=openai_api_key,
                     model=model,
                     temperature=0,
                     n=10,
                     verbose=True)

    agent_executor = create_conversational_retrieval_agent(
        llm, 
        tools, 
        verbose=False, 
        system_message=system_message,
        max_token_limit=3000) # heikel
 
    print(f"\nFrage: {query}")
    with get_openai_callback() as callback:
        answer = agent_executor({"input": query})
        print(f"\nAntwort: {answer['output']}\n\n")
        print(f"Total Tokens: {callback.total_tokens}")
        print(f"Prompt Tokens: {callback.prompt_tokens}")
        print(f"Completion Tokens: {callback.completion_tokens}")
        print(f"Total Cost (USD): ${callback.total_cost}")
    return answer


if __name__ == "__main__":

    QUESTIONS = [
        "Wann bezahlt die EAK jeweils die Rente aus?",
        "Was ist das SECO?",
        "Wer ist Kassenleiterin oder Kassenleiter der EAK?",
    ]

    for question in QUESTIONS:
        OPENAPI_API_KEY = "YOUR_API_KEY"
        SYS_PATH = "YOUR_SYSTEM_PATH"
        ask_agent__eak(question, OPENAPI_API_KEY, SYS_PATH)
