import json
import os
from typing import Dict, List, Any
from langchain.chains.base import Chain
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.vectorstores.base import VectorStore

class VectorSearchChain(Chain):
  query: str
  embedding_engine: str
  embedding_params: Dict[str, Any]
  database: str
  database_params: Dict[str, Any]
  num_results: int
  min_score: float = 0.0
  input_variables: List[str] = []
  output_key: str = 'results'

  @property
  def input_keys(self) -> List[str]:
    return self.input_variables

  @property
  def output_keys(self) -> List[str]:
    return [self.output_key]
  
  def vector_store(self, query) -> VectorStore:    
    if self.database == 'pinecone':
      if 'index' not in self.database_params:
        raise ValueError('Missing index parameter for Pinecone database')

      # import just-in-time so auth will happen after env vars are loaded
      import pinecone
      from langchain.vectorstores.pinecone import Pinecone
      
      index = pinecone.Index(self.database_params['index'])
      return Pinecone(
        index,
        query,
        self.database_params.get('text_key') or 'text',
        namespace=self.database_params.get('namespace') or ''
      )
    else:
      raise ValueError(f'Unknown database: {self.database}')


  def _call(self, inputs: Dict[str, str]) -> Dict[str, str]:
    if self.embedding_engine == 'openai':
      self.embedding_params['openai_api_key'] = os.environ.get("OPENAI_API_KEY")
      embeddings = OpenAIEmbeddings(**self.embedding_params)
    elif self.embedding_engine == 'huggingface':
      model = self.embedding_params.get('model_name') or 'all-mpnet-base-v2'
      embeddings = HuggingFaceEmbeddings(**self.embedding_params, model_name=model)
    else:
      raise ValueError(f'Unknown embedding engine: {self.embedding_engine}')

    vector_store = self.vector_store(embeddings.embed_query)
    
    formatted_query = self.query.format(**inputs)

    items = vector_store.similarity_search_with_score(formatted_query, self.num_results, self.database_params.get('filter'), self.database_params.get('namespace'))

    return {self.output_key: json.dumps([{'text': item[0].page_content, 'meta': item[0].metadata, 'score': item[1]} for item in items if item[1] >= self.min_score])}