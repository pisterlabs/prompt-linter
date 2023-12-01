import os
import modal
import shelve
import logging
import subprocess

MODEL_DIR = "/root/models"

def download_model_to_folder():
    from huggingface_hub import snapshot_download
    
    os.makedirs(MODEL_DIR, exist_ok=True)

    snapshot_download(
        #"meta-llama/Llama-2-13b-chat-hf",
        #"bofenghuang/vigogne-7b-chat",
        "TheBloke/Vigogne-2-7B-Chat-GGUF",
        local_dir=MODEL_DIR,
        token=os.environ["HUGGINGFACE_TOKEN"],
    )
    logging.info(os.listdir("/root/models/"))


image = (
    modal.Image.debian_slim()
    .run_commands("apt-get update")
    .run_commands("apt-get install -y build-essential cmake libopenblas-dev pkg-config")
    .run_commands("pip install --upgrade pip")
    .pip_install("hf-transfer~=0.1")
    .pip_install("huggingface")
    .pip_install("huggingface_hub")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
    .run_function(download_model_to_folder,secret=modal.Secret.from_name("llama-cpp-hf"))
    #.env({"CMAKE_ARGS":"-DLLAMA_CUBLAS=on","FORCE_CMAKE":"1"})
    .env(({"CMAKE_ARGS":"-DLLAMA_BLAS=ON -DLLAMA_BLAS_VENDOR=OpenBLAS"}))
    #.pip_install("llama-cpp-python")
    .run_commands("pip install llama-cpp-python --force-reinstall --upgrade --no-cache-dir")
    #.run_commands("pip install llama-cpp-python --force-reinstall --upgrade --no-cache-dir")
    #.pip_install("ctransformers[cuda]>=0.2.24")
    #.pip_install("ctransformers[cuda]")
    .pip_install("langchain")
)


stub = modal.Stub("llama-cpp-python-inference", image=image)

#@stub.function(gpu="any")
@stub.function()
def install_llama_cpp() :
    from llama_cpp import Llama
    from langchain.llms import LlamaCpp
    """
    from langchain.prompts import ChatPromptTemplate
    from langchain import PromptTemplate, LLMChain
    from langchain.callbacks.manager import CallbackManager
    from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
    """
    #from langchain.llms import CTransformers
    import subprocess
    import time
    start = time.time()
    tstart = start

    #subprocess.run(['nvidia-smi'])
    
    #llm = CTransformers(model='/root/models/vigogne-2-7b-chat.Q4_K_M.gguf',gpu_layers=50, threads=1)
    prompt = """
    <|system|>: Vos réponses sont concises
    <|user|>: {q}
    """
    #llm = LlamaCpp(model_path="/root/models/vigogne-2-7b-chat.Q4_K_M.gguf")
    llm = Llama(model_path="/root/models/vigogne-2-7b-chat.Q4_K_M.gguf")
    
    print("************************ Attention ça va commencer *******************")
    start = time.time()
    tstart = start
    m = "Qui était Henri IV de France ?"
    p = prompt.format(q=m)
    output = llm(p)
    print(output)
    end = time.time()
    print(end - start)
    start = end

    m = "Qui était le Dr Destouches ?"
    p = prompt.format(q=m)
    output = llm(p)
    print(output)
    end = time.time()
    print(end - start)
    start = end

    m = "Comment a débuté la première guerre mondiale ?"
    p = prompt.format(q=m)
    output = llm(p)
    print(output)
    end = time.time()
    print(end - start)
    start = end

    p = prompt.format(q="Comment se mangent les noix ?")
    output = llm(p)
    print(output)
    end = time.time()
    print(end - start)
    start = end

    p = prompt.format(q="Qu'est-ce que le sepuku ?")
    output = llm(p)
    print(output)
    end = time.time()
    print(end - start)
    start = end
    
    #subprocess.run(['nvidia-smi'])
    end = time.time()
    print(end - tstart)

@stub.local_entrypoint()
def main():
    install_llama_cpp.remote()
    

