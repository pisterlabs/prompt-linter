from langchain.chains import LLMChain
from langchain.llms import OpenAI
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
llm_chain = None
def generate_response(prompt):
    global llm_chain
    if not llm_chain:
        initiation_schedule()
    return llm_chain.predict(human_input=prompt)



def initiation_schedule():
    global llm_chain
    from dotenv import load_dotenv
    load_dotenv()
    template = """You are a chatbot having a conversation with a human.\
    {chat_history}\
    Human: {human_input}\
    Chatbot:"""

    prompt = PromptTemplate(
        input_variables=["chat_history", "human_input"], template=template
    )
    memory = ConversationBufferMemory(memory_key="chat_history")

    llm = OpenAI()
    llm_chain = LLMChain(
        llm=llm,
        prompt=prompt,
        verbose=True,
        memory=memory,
    )

