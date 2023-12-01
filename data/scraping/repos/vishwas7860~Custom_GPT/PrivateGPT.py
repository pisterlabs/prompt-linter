import torch
from auto_gptq import AutoGPTQForCausalLM
from huggingface_hub import hf_hub_download
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.llms import HuggingFacePipeline, LlamaCpp
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from constants import CHROMA_SETTINGS, EMBEDDING_MODEL_NAME, PERSIST_DIRECTORY, MODEL_ID, MODEL_BASENAME

# from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.vectorstores import Chroma
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    GenerationConfig,
    LlamaForCausalLM,
    LlamaTokenizer,
    pipeline,
)

def load_model(model_id, model_basename):
    
    model_path = hf_hub_download(repo_id=model_id, filename=model_basename)
    max_ctx_size = 2048
    kwargs = {
        "model_path": model_path,
        "n_ctx": max_ctx_size,
        "max_tokens": max_ctx_size,
    }
    
    kwargs["n_gpu_layers"] = 1000
    kwargs["n_batch"] = max_ctx_size
    return LlamaCpp(**kwargs)


def main():

    embeddings = HuggingFaceInstructEmbeddings(model_name=EMBEDDING_MODEL_NAME, model_kwargs={"device": "cuda"})

    db = Chroma(
        persist_directory=PERSIST_DIRECTORY,
        embedding_function=embeddings,
        client_settings=CHROMA_SETTINGS,
    )
    retriever = db.as_retriever()
    

    template = """Use the following pieces of context to answer the question at the end. If you don't know the answer,\
    just say that you don't know, don't try to make up an answer.

    {context}

    {history}
    Question: {question}
    Helpful Answer:"""

    prompt = PromptTemplate(input_variables=["history", "context", "question"], template=template)
    memory = ConversationBufferMemory(input_key="question", memory_key="history")

    llm = load_model(model_id=MODEL_ID, model_basename=MODEL_BASENAME)

    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt, "memory": memory},
    )
   
    while True:
        query = input("Enter Query (for exit type 'exit')\n")
        # Get the answer from the chain
        if query == "exit":
          print("Thanks for reaching me!!!!!!!")
          break
        else:
          res = qa(query)

          # Print the result
          print("\n\n> Question:")
          print(query)
          print("\n> Answer:")
          print(res["result"])


if __name__ == "__main__":
    main()
