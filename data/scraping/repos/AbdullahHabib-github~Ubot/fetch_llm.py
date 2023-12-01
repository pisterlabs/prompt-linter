import os

def replicate_llm():
    from langchain.llms import Replicate
    with open("replicate_token.txt", 'r') as file:
        for line in file:
            RTOKEN= (line)
    os.environ["REPLICATE_API_TOKEN"] = RTOKEN
    llm = Replicate(model="lucataco/llama-2-7b-chat:6ab580ab4eef2c2b440f2441ec0fc0ace5470edaf2cbea50b8550aec0b3fbd38", model_kwargs={"temperature": 0, "max_new_tokens": 128})
    return llm


def hf_model():
    from langchain import HuggingFaceHub
    with open("hugging_face_token.txt", 'r') as file:
        for line in file:
            HTOKEN= (line)
    os.environ["HUGGINGFACEHUB_API_TOKEN"] = HTOKEN
    return HuggingFaceHub(repo_id="google/flan-t5-small", model_kwargs={"temperature": 0, "max_length": 512})




def Llama():
    from llamaapi import LlamaAPI
    from langchain_experimental.llms import ChatLlamaAPI
    with open("Api_keys/llama_api.txt", 'r') as file:
            for line in file:
                api = (line)

    llama = LlamaAPI('LL-'+api)
    chat_llama = ChatLlamaAPI(client=llama)
    return chat_llama

