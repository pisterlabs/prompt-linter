# jank local frontend for llm models 

# libraries 
from colorama import Fore
from langchain import LlamaCpp, ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

# init model
user_input = ""
callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
conversation_buffer = ConversationBufferMemory()
ctx_size = 4096
ggml_model = LlamaCpp(
    model_path = "./based-30b.ggmlv3.q4_K_M.bin", 
    n_ctx = ctx_size, 
    callback_manager = callback_manager, 
    verbose = False)

print (Fore.GREEN + "model name")
# run that shit
while user_input != "exit":
    user_input = input(Fore.BLUE + "> ")
    generate = ConversationChain(
        llm = ggml_model,
        memory = conversation_buffer,
        verbose = False)
    # output response
    print (Fore.RED)
    generate.predict(input = user_input)
    print("\n")
