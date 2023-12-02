from langchain.embeddings import HuggingFaceEmbeddings  # Імпорт бібліотеки для роботи з векторними представленнями тексту
from langchain.vectorstores import Chroma  # Імпорт бібліотеки для зберігання та роботи з векторними представленнями тексту
from langchain import HuggingFaceHub  # Імпорт бібліотеки, яка використовує модель
from langchain.chains import ConversationalRetrievalChain  # Імпорт бібліотеки для створення ланцюжка обробки чат-взаємодії
from langchain.memory import ConversationBufferMemory  # Імпорт бібліотеки для роботи з пам'яттю для зберігання історії розмови

# Імпорт конфігураційних налаштувань з файлу config.py
from src.conf.config import settings


def create_conversation() -> ConversationalRetrievalChain:
    # Встановлення шляху до папки для збереження даних
    persist_directory = settings.db_dir
    repo_id = "tiiuae/falcon-7b-instruct"
    # Ініціалізація моделі для text-generation
    llm_model = HuggingFaceHub(huggingfacehub_api_token=settings.hf_api_key,
                           repo_id=repo_id,
                           model_kwargs={"temperature": 0.6,
                                         "max_new_tokens": 100})

    # Ініціалізація моделі для створення векторних представлень тексту
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

    # Ініціалізація бази даних для зберігання векторних представлень тексту
    db = Chroma(
        persist_directory=persist_directory,
        embedding_function=embeddings
    )

    # Ініціалізація пам'яті для зберігання історії розмови
    memory = ConversationBufferMemory(
        memory_key='chat_history',
        return_messages=False
    )

    # Створення ланцюжка обробки чат-взаємодії з використанням моделі llm
    qa = ConversationalRetrievalChain.from_llm(
        llm=llm_model,
        chain_type='stuff',
        retriever=db.as_retriever(),
        memory=memory,
        get_chat_history=lambda h: h,
        verbose=True
    )

    return qa
