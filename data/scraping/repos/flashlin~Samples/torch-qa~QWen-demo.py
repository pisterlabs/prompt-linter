from langchain.llms import LlamaCpp
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.memory import ConversationSummaryMemory
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig

model_path = './models/Qwen-14B-Chat-Int4'

# callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
# llm = LlamaCpp(
#     model_path=model_path,
#     n_ctx=5000,
#     n_gpu_layers=40,
#     n_threads=15,
#     n_batch=512,
#     f16_kv=True,
#     callback_manager=callback_manager,
#     verbose=True,
# )
#
# # memory = ConversationSummaryMemory(llm=llm, memory_key="chat_history", return_messages=True)
# #response, history = llm.chat(tokenizer, "你好", history=None)
# response = llm('你好')
# print(response)


tokenizer = AutoTokenizer.from_pretrained(
    model_path, trust_remote_code=False, resume_download=False,
)

device_map = "auto"
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    device_map=device_map,
    trust_remote_code=False,
    resume_download=False,
).eval()

config = GenerationConfig.from_pretrained(
    model_path, trust_remote_code=False, resume_download=False,
)

query = "use C# write Hello"
resp = model.chat_stream(tokenizer, query, history=[], generation_config=config)
print(f"{resp=}")

