import os

#LLM
from langchain.llms import LlamaCpp
from langchain.schema import SystemMessage
from langchain.prompts import PromptTemplate, ChatPromptTemplate, HumanMessagePromptTemplate, MessagesPlaceholder

#Prompt and Memory
from agent.prompts import SYS_PROMPT
from langchain.memory import ConversationBufferMemory

#Chain
from langchain.chains import LLMChain

#Callback para Streaming
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

from utils import download_model

OPENAI_API_KEY = os.getenv('OPENAI_APIKEY')

# Encontra o caminho atual do script



LLM_MODEL = "llama-2-7b-chat.Q4_K_M.gguf"
# Concatena com o caminho relativo do modelo

MODEL_PATH = download_model("TheBloke/Llama-2-7b-Chat-GGUF","llama-2-7b-chat.Q4_K_M.gguf")
               

#Define o LLM

# Callbacks support token-wise streaming

callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])

llm = LlamaCpp(
    model_path=MODEL_PATH,
    temperature=0.75,
    max_tokens=512,
    top_p=1,
    callback_manager=callback_manager, 
    verbose=True, # Verbose is required to pass to the callback manager
    streaming=True,
    stop=["Human:"]
)
#Memória
prompt = PromptTemplate.from_template("<s>[INST] <<SYS>>\n" + SYS_PROMPT + "\n<</SYS>>\n\n{human_input} [/INST]</s>")
    
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

chat_llm_chain = LLMChain(
    llm=llm,
    prompt=prompt,
    memory=memory,
    verbose=True
)