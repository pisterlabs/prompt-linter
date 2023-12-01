import os
from fastapi import FastAPI, Header
from pydantic import BaseModel
from datetime import datetime
from typing import Optional

from llama_cpp import Llama
import modal
import shelve
import logging
import subprocess

from pydantic import BaseModel

MODEL_DIR = "/root/models"

def download_model_to_folder():
    os.makedirs(MODEL_DIR, exist_ok=True)

    subprocess.run(['huggingface-cli',
                    'download',
                    'TheBloke/Vigogne-2-7B-Chat-GGUF',
                    'vigogne-2-7b-chat.Q5_K_M.gguf', 
                    '--local-dir',
                    MODEL_DIR,
                    '--local-dir-use-symlinks',
                    'False'])
  
    logging.info(os.listdir("/root/models/"))



image = (
    modal.Image.debian_slim()
    .run_commands("apt-get update")
    .run_commands("apt-get install -y build-essential cmake libopenblas-dev pkg-config")
    .run_commands("pip install --upgrade pip")
    .pip_install("torch")
    .pip_install("huggingface")
    .pip_install("huggingface_hub")
    .run_function(download_model_to_folder,secret=modal.Secret.from_name("llama-cpp-hf"))
    .run_commands("pip install llama-cpp-python --force-reinstall --upgrade --no-cache-dir")
    .pip_install("langchain")
)


stub = modal.Stub("llama-cpp-python-nu", image=image)

@stub.function()
def install_llama_cpp() :
    import torch
    from llama_cpp import Llama
    from langchain.llms import LlamaCpp
    """
    from langchain.prompts import ChatPromptTemplate
    from langchain import PromptTemplate, LLMChain
    from langchain.callbacks.manager import CallbackManager
    from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
    """

    import subprocess
    import time
    start = time.time()
    tstart = start

    if torch.cuda.is_available():
        print("GPU is available")
    else:
        print("GPU is not available")

    prompt = """
    <|system|>: Vos réponses sont concises
    <|user|>: {q}
    """
    #llm = LlamaCpp(model_path="/root/models/vigogne-2-7b-chat.Q5_K_M.gguf")
    llm = Llama(model_path="/root/models/vigogne-2-7b-chat.Q5_K_M.gguf")
    

@stub.local_entrypoint()
def main():
    install_llama_cpp.remote()
    

ex = """
<s>[INST] <<SYS>>
Vous êtes Vigogne, un assistant IA créé par Zaion Lab. Vous suivez extrêmement bien les instructions. 
Aidez autant que vous le pouvez.
<</SYS>>
Bonjour ! Comment ça va aujourd'hui ? [/INST] 
Bonjour ! Je suis une IA, donc je n'ai pas de sentiments, mais je suis prêt à vous aider. Comment puis-je vous assister aujourd'hui ? </s>
[INST] Quelle est la hauteur de la Tour Eiffel ? [/INST] 
La Tour Eiffel mesure environ 330 mètres de hauteur. </s>
[INST] Comment monter en haut ? [/INST]"""
@stub.cls()
class Model:
    def __enter__(self):
        self.prompt = """
            <s>[system]<<SYS>> Vos réponses sont concises<<SYS>>
            {c}
            [INST]{q}[/INST]
            """

        self.llm =Llama(model_path="/root/models/vigogne-2-7b-chat.Q5_K_M.gguf",n_ctx=4096)
        # Load the model. Tip: MPT models may require `trust_remote_code=true`.

    @modal.method()
    def generate(self, user_question, context=""):
        p = self.prompt.format(q=user_question,c=context)
        
        logging.info(context+" "+user_question)
        result = self.llm.create_chat_completion(messages=[p],max_tokens=2048)
        
        logging.info(result)
        return result

gmodel = Model()


web_app = FastAPI()

"""class Item(BaseModel):
    prompt: str
    context: str"""

class TransactionType(BaseModel):
    transaction_id: str
    type: str
    timestamp_requête: datetime
    timestamp_réponse: Optional[datetime] = None
    question: Optional[str] = None  # Si nécessaire
    réponse: Optional[str] = None
    contexte: Optional[str] = None  # Si nécessaire

@web_app.get("/")
async def handle_root(user_agent: str = Header(None)):
    logging.info(f"GET /     - received user_agent={user_agent}")
    return "Hello World"


@web_app.post("/question")
async def handle_question(transaction: TransactionType):
    logging.info(
        f"POST /question - received transaction.prompt={transaction.qestion}, transaction.context={transaction.context}"
    )
    # logging.info(item)
    # if item.context is not None:
    #     # Logique si context est fourni
    #     answer = gmodel.generate.remote(item.prompt, item.context)
    # else:
    #     # Logique si context n'est pas fourni
    #     answer = gmodel.generate.remote(item.prompt)
    # return answer

    # Générer la réponse
    if transaction.context is not None:
        answer = await gmodel.generate.remote(transaction.question, transaction.contexte)
    else:
        answer = await gmodel.generate.remote(transaction.question)

    # Mettre à jour la transaction avec la réponse
    transaction.réponse = answer
    transaction.timestamp_réponse = datetime.now()

    # Convertir l'objet Pydantic en dictionnaire pour le retourner
    return transaction.dict()

@stub.function(image=image)
@modal.asgi_app()
def fastapi_app():
    return web_app


if __name__ == "__main__":
    stub.deploy("webapp")
