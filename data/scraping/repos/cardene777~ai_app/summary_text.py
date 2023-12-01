import os
import openai
from llama_index import StorageContext, StringIterableReader, GPTVectorStoreIndex, load_index_from_storage
from dotenv import load_dotenv

load_dotenv()
openai.api_key = os.getenv('OPENAI_API_KEY')
STORAGE_PATH = "./storage"


def get_summary_text(audio_text: str, summary_prompt: str) -> str:
    if not os.path.exists(STORAGE_PATH):
        os.makedirs(STORAGE_PATH)
    try:
        storage_context = StorageContext.from_defaults(persist_dir=STORAGE_PATH)
        vector_index = load_index_from_storage(storage_context)
    except Exception:
        documents = StringIterableReader().load_data(texts=[audio_text])
        vector_index = GPTVectorStoreIndex.from_documents(documents)
        vector_index.storage_context.persist(persist_dir=STORAGE_PATH)

    query_engine = vector_index.as_query_engine()
    response = query_engine.query(summary_prompt)

    return response.response


def main():
    text = """
    ...
    """
    prompt = "テキストの内容を5~10の項目に分けて要約してください。"
    summary_text = get_summary_text(text, prompt)
    print(f"summary_text: {summary_text}")


if __name__ == "__main__":
    main()
