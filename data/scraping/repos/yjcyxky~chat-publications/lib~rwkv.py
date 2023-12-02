import os
from langchain.llms import RWKV
from llama_index import (LLMPredictor, ServiceContext,
                         LangchainEmbedding, PromptHelper)

from llama_index.langchain_helpers.text_splitter import TokenTextSplitter
from llama_index.node_parser.simple import SimpleNodeParser
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
    

def get_service_context(max_chunk_overlap=0.5, max_input_size=2000, 
                        num_output=512, chunk_size_limit=512) -> ServiceContext:
    os.environ["MAX_INPUT_SIZE"] = str(max_input_size)

    llm_predictor = LLMPredictor(
        llm=RWKV(model="/data/rwkv-14b-v11/RWKV-4-Raven-14B-v11x-Eng99-Other1-20230501-ctx8192.pth", 
                     strategy="cuda fp16", tokens_path="/data/rwkv-14b-v11/20B_tokenizer.json",
                     max_tokens_per_generation=512)
    )

    embed_model = LangchainEmbedding(HuggingFaceEmbeddings())

    node_parser = SimpleNodeParser(text_splitter=TokenTextSplitter(
        chunk_size=chunk_size_limit, chunk_overlap=max_chunk_overlap)
    )
    prompt_helper = PromptHelper(max_input_size, num_output, max_chunk_overlap)
    service_context = ServiceContext.from_defaults(
        llm_predictor=llm_predictor, embed_model=embed_model,
        prompt_helper=prompt_helper, node_parser=node_parser,
        chunk_size_limit=chunk_size_limit
    )
    return service_context
