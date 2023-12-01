from dive.util.configAPIKey import set_openai_api_key_from_env, set_hugging_face_auth_from_env
import environ
env = environ.Env()
environ.Env.read_env()

import os
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain import OpenAI


def get_llm():
    OPENAI_API_KEY = env.str('OPENAI_API_KEY', default='') or os.environ.get('OPENAI_API_KEY', '')

    llm = env.str('LLM', default='')

    if llm and llm == 'llama2':
        from huggingface_hub import hf_hub_download
        from langchain.embeddings import LlamaCppEmbeddings
        hf_auth = env.str('HUGGING_FACE_AUTH', default='') or os.environ.get('HUGGING_FACE_AUTH', '')
        set_hugging_face_auth_from_env(hf_auth)
        model_path = hf_hub_download(repo_id='TheBloke/Llama-2-7B-GGML', filename='llama-2-7b.ggmlv3.q5_1.bin',
                                     use_auth_token=hf_auth)
        from langchain.llms import LlamaCpp
        return LlamaCpp(
            model_path=model_path,
            input={"temperature": 0, "max_length": 2000, "top_p": 1},
            verbose=True,
        )

    elif OPENAI_API_KEY:
        set_openai_api_key_from_env(OPENAI_API_KEY)
        return OpenAI(temperature=0)

    return None


def get_embeddings():
    OPENAI_API_KEY = env.str('OPENAI_API_KEY', default='') or os.environ.get('OPENAI_API_KEY', '')

    embeddings = env.str('EMBEDDINGS', default='')

    if embeddings and embeddings == 'llama2':
        from huggingface_hub import hf_hub_download
        from langchain.embeddings import LlamaCppEmbeddings
        hf_auth = env.str('HUGGING_FACE_AUTH', default='') or os.environ.get('HUGGING_FACE_AUTH', '')
        set_hugging_face_auth_from_env(hf_auth)
        model_path = hf_hub_download(repo_id='TheBloke/Llama-2-7B-GGML', filename='llama-2-7b.ggmlv3.q5_1.bin',
                                     use_auth_token=hf_auth)
        return LlamaCppEmbeddings(model_path=model_path)

    elif OPENAI_API_KEY:
        set_openai_api_key_from_env(OPENAI_API_KEY)
        return OpenAIEmbeddings()

    return None