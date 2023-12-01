from langchain.agents import initialize_agent
from langchain.agents.types import AgentType
from langchain.tools import Tool
from langchain.chains import RetrievalQA
from langchain.memory import ConversationBufferMemory
from config import OPENAI_API_KEY
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredExcelLoader
from langchain.vectorstores import Chroma
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
llm = ChatOpenAI(temperature=0)

# founder


def embed_founder():
    persist_directory = "db_founder"
    return Chroma(embedding_function=OpenAIEmbeddings(),
                  persist_directory=persist_directory)

# Template for the model one


def template_founder():
    template = """You are ConnectToLearn a dedicated AI interface of Plaksha University, entrusted with the task of seamlessly connecting users who harbor specific academic interests or fields of inquiry with founders possessing expert knowledge in corresponding domains. Your role encompasses not only bridging this informational gap but also providing insightful answers concerning various founders.

    When a user inputs a particular field, you possess the capability to suggest a founder who specializes in that specific field. Your knowledge is rooted in the context outlined below, which serves as the foundation for your responses, always go through the context thoroughly before answering any question:

    {context}

    Should you encounter questions for which you lack information, don't hesitate to communicate your limitations and let the user know that you are unable to furnish an appropriate response. In scenarios where you're requested to recommend a founder in a given field, ensure you explore multiple options whenever available. However, if a solitary founder stands as the sole authority in that domain, it's acceptable to provide their name as the definitive reference.

    Moreover, your expertise extends to suggesting a founder whose knowledge closely aligns with the user's query, in cases where an exact match is absent. This functionality enhances your ability to cater to a wider array of user needs.

    With this comprehensive understanding and enhanced functionality, you're prepared to address the user's inquiry adeptly:
    Question: {question}
    Helpful Answer:"""

    QA_CHAIN_PROMPT = PromptTemplate(
        input_variables=["context", "question"], template=template,)
    return QA_CHAIN_PROMPT


def retrieval_founder(QA_CHAIN_PROMPT, vectorstore):
    qa_chain = RetrievalQA.from_chain_type(llm,
                                           retriever=vectorstore.as_retriever(),
                                           chain_type_kwargs={"prompt": QA_CHAIN_PROMPT})
    return qa_chain


def memory():
    conversational_memory = ConversationBufferMemory(
        memory_key="chat_history", return_messages=True
    )
    return conversational_memory


# professor

def embed_prof():
    persist_directory = "db"
    return Chroma(embedding_function=OpenAIEmbeddings(),
                  persist_directory=persist_directory)

# Template for the model one


def template_prof():
    template = """You are the dedicated AI interface of Plaksha University, entrusted with the task of seamlessly connecting users who harbor specific academic interests or fields of inquiry with professors possessing expert knowledge in corresponding domains. Your role encompasses not only bridging this informational gap but also providing insightful answers concerning various professors.

    When a user inputs a particular field, you possess the capability to suggest a professor who specializes in that specific field. Your knowledge is rooted in the context outlined below, which serves as the foundation for your responses always go through the context thoroughly before answering any question:

    {context}

    Should you encounter questions for which you lack information, don't hesitate to communicate your limitations and let the user know that you are unable to furnish an appropriate response. In scenarios where you're requested to recommend a professor in a given field, ensure you explore multiple options whenever available. However, if a solitary professor stands as the sole authority in that domain, it's acceptable to provide their name as the definitive reference.

    Moreover, your expertise extends to suggesting a professor whose expertise closely aligns with the user's query, in cases where an exact match is absent. This functionality enhances your ability to cater to a wider array of user needs.

    With this comprehensive understanding and enhanced functionality, you're prepared to address the user's inquiry adeptly:
    Question: {question}
    Helpful Answer:"""
    QA_CHAIN_PROMPT = PromptTemplate(
        input_variables=["context", "question"], template=template,)
    return QA_CHAIN_PROMPT


def retrieval_prof(QA_CHAIN_PROMPT, vectorstore):
    qa_chain = RetrievalQA.from_chain_type(llm,
                                           retriever=vectorstore.as_retriever(),
                                           chain_type_kwargs={"prompt": QA_CHAIN_PROMPT})
    return qa_chain


def question_factory(founder_info, prof_info, conversational_memory):
    system_message = """
           "Your name is connect to learn"
           "You are a Virtual Assistant to Plaksha University"
            "You job is to recommend a founder or professor based on the question asked by the user and you can answer question about a professor of founder"
            "Your action input must always be the question which I asked you"
            "Ask follow up question if you do not understand the questions being asked"
            "Note all founders and professors are always available"
            "You should only talk within the context of problem."
            """
    tools = [
        Tool(
            name="prof_info",
            func=prof_info.run,
            description="Use this to answer question related to professor or recommendation of professor",
        ),
        Tool(
            name="founder_info",
            func=founder_info.run,
            description="Use this to answer question related to founder or recommendation of founder",
        )
    ]
    executor = initialize_agent(
        agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
        tools=tools,
        llm=llm,
        memory=conversational_memory,
        agent_kwargs={"system_message": system_message},
        verbose=True,
        handle_parsing_errors=True,
    )
    return executor


def main():
    # embed and store for prif
    vectorstore_prof = embed_prof()

    # prompt for prof
    QA_CHAIN_PROMPT_prof = template_prof()

    # setting up first retrieval for prof
    qa_chain = retrieval_prof(QA_CHAIN_PROMPT_prof, vectorstore_prof)

    # embed and store
    vectorstore_founder = embed_founder()

    # prompt
    QA_CHAIN_PROMPT_founder = template_founder()

    # setting up first retrieval
    founder_info = retrieval_founder(
        QA_CHAIN_PROMPT_founder, vectorstore_founder)

    # running the first input
    executor = question_factory(founder_info, qa_chain, memory())

    # Loop
    while True:
        user_input = input("Input your question \n")
        answer = executor.run(user_input)
        print(answer)


if __name__ == "__main__":
    main()
