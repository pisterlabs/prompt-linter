from langchain.llms import LlamaCpp
from torch import cuda

print(cuda.current_device())

model_path = r'llama-2-7b-chat-codeCherryPop.ggmlv3.q2_K.bin'

llm = LlamaCpp(
    model_path=model_path,
    n_gpu_layers=4,
    n_ctx=512,
    temperature=0
)

output = llm("Q: Name the planets in the solar system? A: ", max_tokens=32, stop=["Q:", "\n"])

print(output)