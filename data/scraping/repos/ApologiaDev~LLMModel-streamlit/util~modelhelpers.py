
from langchain import HuggingFaceHub
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceEmbeddings
from langchain.embeddings.fake import FakeEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter


def get_llm_model(config):
    llm_config = config['llm']
    hub = llm_config.get('hub', 'openai')
    if hub == 'openai':
        model = llm_config.get('model', 'gpt-3.5-turbo')
        temperature = config.get('temperature', 0.7)
        max_tokens = config.get('max_tokens', 600)
        return ChatOpenAI(temperature=temperature, model_name=model, max_tokens=max_tokens)
    elif hub == 'huggingface':
        model = llm_config.get('model', 'google/flan-t5-xxl')
        model_kwargs = llm_config.get('model_kwargs')  # example: {'temperature': temperature, 'max_length': max_tokens}
        if model_kwargs is None:
            return HuggingFaceHub(repo_id=model)
        else:
            return HuggingFaceHub(repo_id=model, model_kwargs=model_kwargs)
    else:
        raise ValueError('Unknown LLM specified!')


def get_embeddings_model(config):
    embedding_config = config.get('embedding')
    if embedding_config is None:
        return FakeEmbeddings(size=768)
    hub = embedding_config.get('hub')
    if (hub is None) or (hub == 'openai'):
        return OpenAIEmbeddings()
    elif hub == 'huggingface':
        model = embedding_config.get('model', 'sentence-transformers/all-MiniLM-L6-v2')
        model_kwargs = embedding_config.get('model_kwargs')
        embeddings_model = HuggingFaceEmbeddings(model_name=model) if model_kwargs is None else HuggingFaceEmbeddings(model_name=model, model_kwargs=model_kwargs)
        if embeddings_model.client.tokenizer.pad_token is None:
            embeddings_model.client.tokenizer.pad_token = embeddings_model.client.tokenizer.eos_token
        return embeddings_model


text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=900,
    chunk_overlap=20,
    length_function=len
)
