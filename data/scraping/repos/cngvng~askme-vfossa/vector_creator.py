from backend.vectorstore.pineconeDB import PineconeDB
import openai
from backend.core.settings import settings


class VectorCreator:
    openai.api_key = settings.API_KEY
    vectorstores = {
        'pinecone': PineconeDB
    }

    @classmethod
    def create_vectorstore(cls, type, *args, **kwargs):
        vectorstore_class = cls.vectorstores.get(type.lower())
        if not vectorstore_class:
            raise ValueError(f"No vectorstore class found for type {type}")
        return vectorstore_class(*args, **kwargs)