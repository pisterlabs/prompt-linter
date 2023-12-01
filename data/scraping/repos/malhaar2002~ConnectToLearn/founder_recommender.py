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
llm = ChatOpenAI(model_name="gpt-4", temperature=0)


def embed():
    persist_directory = "db_founder"
    return Chroma(embedding_function=OpenAIEmbeddings(),
                  persist_directory=persist_directory)

# Template for the model one


def template_1():
    template = """You are ConnectToLearn owned by Plaksha University. You help with connecting user who have a particular interest in a topic to a Founder who have experience in such topic. You should give founders name and information about them based on the field of interest of the student in the Question asked by the student. You also can answer questions regarding a founder. When you are asked about a founder, ensure you provide a well detailed answer to it. 
    Ensure you go through the below context before answering a question, if you do not know the answer just tell the user that you do not have any information about that.
    You can only get the answer from the context:-
    {context}

    Question: {question}
    Helpful Answer:"""
    QA_CHAIN_PROMPT = PromptTemplate(
        input_variables=["context", "question"], template=template,)
    return QA_CHAIN_PROMPT


def retrieval(QA_CHAIN_PROMPT, vectorstore):
    qa_chain = RetrievalQA.from_chain_type(llm,
                                           retriever=vectorstore.as_retriever(),
                                           chain_type_kwargs={"prompt": QA_CHAIN_PROMPT})
    return qa_chain
# memory


def memory():
    conversational_memory = ConversationBufferMemory(
        memory_key="chat_history", return_messages=True
    )
    return conversational_memory


def question_factory(founder_info_1, conversational_memory):
    system_message = """
            "You are a Virtual Assistant to Plaksha University"
            "You job is to recommend a founder or professor based on the question asked by the user and you can answer question about a professor of founder"
            "Your action input must always be the question which I asked you"
            "Note all founders and professors are always available"
            "You should only talk within the context of problem."
            """
    tools = [
        Tool(
            name="qa_founder_1",
            func=founder_info_1.run,
            description="The Tool to answer all the questions",
        )
    ]
    executor = initialize_agent(
        agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
        tools=tools,
        llm=llm,
        memory=conversational_memory,
        agent_kwargs={"system_message": system_message},
        verbose=True,
    )
    return executor


def main():
    # embed and store
    vectorstore = embed()

    # prompt
    QA_CHAIN_PROMPT = template_1()

    # setting up first retrieval
    founder_info = retrieval(QA_CHAIN_PROMPT, vectorstore)
    executor = question_factory(founder_info, memory())

    # Loop
    while True:
        user_input = input("Input your question \n")
        answer = executor.run(user_input)
        print(answer)


if __name__ == "__main__":
    main()
