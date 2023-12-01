# import dependencies
from langchain.llms import LlamaCpp
# from llama_cpp import Llama

# Load the LLM
# llm = Llama(model_path="./models/llama-7b.ggmlv3.q3_K_L.bin")
llm = LlamaCpp(model_path="models/llama-2-7b-chat.Q2_K.gguf")

# Pass a prompt to the LLM
response = llm("Who directed the Dark Knight?")

# Check the response
print(response)
