from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, ConversationChain
from langchain.memory import ConversationBufferMemory, CombinedMemory, ConversationSummaryMemory
from dotenv import load_dotenv
load_dotenv()

def create_conversation(template):
    conv_memory = ConversationBufferMemory(
        memory_key="chat_history_lines",
        input_key="input"
    )
    summary_memory = ConversationSummaryMemory(llm=OpenAI(), input_key="input")
    memory = CombinedMemory(memories=[conv_memory, summary_memory])
    PROMPT = PromptTemplate(
        input_variables=["history","input","chat_history_lines"],
        template=template,
    )
    conversation = ConversationChain(
        llm=OpenAI(temperature=0.7,frequency_penalty=0.7, presence_penalty=0.6),
        verbose=True,
        memory=memory,
        prompt=PROMPT
    )
    return conversation




