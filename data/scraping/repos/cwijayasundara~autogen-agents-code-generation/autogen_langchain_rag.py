import autogen

from dotenv import load_dotenv
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
from langchain.memory import ConversationBufferMemory
from langchain.llms import OpenAI
from langchain.chains import ConversationalRetrievalChain

config_list = autogen.config_list_from_json(
    "OAI_CONFIG_LIST.json",
    filter_dict={
        "model": ["gpt-3.5-turbo-16k"],
    },
)

# Get API key
load_dotenv()
print(config_list)

loaders = [PyPDFLoader('rag/docs/llava.pdf')]
docs = []

for doc in loaders:
    docs.extend(doc.load())

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000)
docs = text_splitter.split_documents(docs)

vectorstore = Chroma(
    collection_name="research_papers",
    embedding_function=OpenAIEmbeddings()
)
vectorstore.add_documents(docs)

qa = ConversationalRetrievalChain.from_llm(
    OpenAI(temperature=0),
    vectorstore.as_retriever(),
    memory=ConversationBufferMemory(memory_key="chat_history", return_messages=True)
)

result = qa(({"question": "What is llava?"}))
print(result)


def answer_query_from_vector_db(question):
    response = qa({"question": question})
    return response["answer"]


llm_config = {
    "request_timeout": 600,
    "seed": 42,
    "config_list": config_list,
    "temperature": 0,
    "functions": [
        {
            "name": "answer_query_from_vector_db",
            "description": "Answer a question based on a vector database",
            "parameters": {
                "type": "object",
                "properties": {
                    "question": {
                        "type": "string",
                        "description": "The question to ask in relation to LLaVA",
                    }
                },
                "required": ["question"],
            },
        }
    ],
}

# create an AssistantAgent instance named "assistant"
assistant = autogen.AssistantAgent(
    name="assistant",
    system_message="I'm an assistant that can answer questions about LLaVA.  I will not be involved in future "
                   "conversations",
    llm_config=llm_config,
)
# create a UserProxyAgent instance named "user_proxy"
user_proxy = autogen.UserProxyAgent(
    name="user_proxy",
    human_input_mode="NEVER",
    max_consecutive_auto_reply=10,
    code_execution_config={"work_dir": "code"},
    llm_config=llm_config,
    system_message="""Reply TERMINATE if the task has been solved at full satisfaction.
Otherwise, reply CONTINUE, or the reason why the task is not solved yet.""",
    function_map={"answer_query_from_vector_db": answer_query_from_vector_db}
)

user_proxy.initiate_chat(
    assistant,
    message="""I'm writing a blog to introduce LLaVA. Find answers to the 3 questions 
    below and write a summery.
    1. What is LLaVA?
    2. Why do you need it?
    3. How to use?
    """
)
