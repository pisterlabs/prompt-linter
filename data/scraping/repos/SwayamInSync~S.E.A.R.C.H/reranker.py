from llama_index.indices.postprocessor.cohere_rerank import CohereRerank

from config import config

# cohere rerank
api_key = config.cohere_api
cohere_rerank = CohereRerank(api_key=api_key, top_n=config.reranker_top_k)

# more rerankers are yet to add
