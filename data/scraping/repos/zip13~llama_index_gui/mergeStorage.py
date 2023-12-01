from llama_index import SimpleDirectoryReader, GPTListIndex,ServiceContext,StorageContext, GPTVectorStoreIndex, LLMPredictor, PromptHelper,load_index_from_storage
from langchain import OpenAI

import sys
from env import ini_env


def load_index(persist_dir):
     # LLM Predictor (gpt-3.5-turbo)
    max_input_size = 4096

    num_outputs = 512

    max_chunk_overlap = 20

    chunk_size_limit = 600

    prompt_helper = PromptHelper(max_input_size, num_outputs, max_chunk_overlap, chunk_size_limit=chunk_size_limit)

    llm_predictor = LLMPredictor(llm=OpenAI(temperature=0.3, model_name="gpt-3.5-turbo", max_tokens=num_outputs))

    service_context = ServiceContext.from_defaults(
                llm_predictor=llm_predictor,
                prompt_helper=prompt_helper,
                chunk_size_limit=chunk_size_limit
                )
    # rebuild storage context
    storage_context = StorageContext.from_defaults(persist_dir=persist_dir)
    # load index
    index = load_index_from_storage(storage_context,service_context=service_context)
    return index


ini_env()
#加载已有index向量库
index5g = load_index('./storage/5g')

#加载新增文档
documents = SimpleDirectoryReader('./docs/bai').load_data()

#添加文档到已有index
for doc in documents:
    index5g.insert(doc)

#保存到新的位置
index5g.storage_context.persist(persist_dir='./storage/all')