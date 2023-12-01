from langchain.llms import LlamaCpp

MODEL_PATH = "models/TheBloke_Mistral-7B-Instruct-v0.1-GGUF/mistral-7b-instruct-v0.1.Q4_K_M.gguf"

llm = LlamaCpp(
    model_path=MODEL_PATH,
    n_ctx=5000,
    n_gpu_layers=40,
    n_threads=15,
    n_batch=512,
    f16_kv=True,
    #callback_manager=callback_manager,
    verbose=True,
)

text = "<s>[INST] What is your favourite condiment? [/INST]"
"Well, I'm quite partial to a good squeeze of fresh lemon juice. It adds just the right amount of zesty flavour to whatever I'm cooking up in the kitchen!</s> "
"[INST] Do you have mayonnaise recipes? [/INST]"
resp = llm("use C# code write Hello")
print(f"{resp=}")
