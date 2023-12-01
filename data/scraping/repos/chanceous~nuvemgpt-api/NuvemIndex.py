import os

from langchain import OpenAI
from llama_index import ServiceContext, LLMPredictor, PromptHelper, StorageContext, load_index_from_storage, \
    GPTVectorStoreIndex, SimpleDirectoryReader

class NuvemIndex():
    def __init__(self):
        pass

    def query(text):
        storage_context = StorageContext.from_defaults(persist_dir=os.getenv('INDEX_STORAGE'))
        index = load_index_from_storage(storage_context)
        query_engine = index.as_query_engine()
        response = query_engine.query(text)

        return response.response

    def indexDocs(self):
        max_input_size = 4096
        num_outputs = 512

        prompt_helper = PromptHelper(max_input_size, num_outputs, chunk_overlap_ratio=0.1)

        llm_predictor = LLMPredictor(llm=OpenAI(temperature=0.7, model_name="text-davinci-003"))
        service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor, prompt_helper=prompt_helper)

        documents = SimpleDirectoryReader('./training').load_data()
        index = GPTVectorStoreIndex.from_documents(
            documents, service_context=service_context
        )

        index.storage_context.persist(persist_dir=os.getenv('INDEX_STORAGE'))

        print('Datos indexados correctamente')
