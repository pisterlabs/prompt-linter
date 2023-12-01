from langchain.prompts import PromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough
from lib.const import CONFIG_BASIC_MODEL, CONFIG_SF
from lib.datasources.utils import get_datasources
from lib.model_manager import ModelManager
from lib.vectorstore import VectorStore


CLASSIFICATION_PROMPT = """Classify the question into salesforce or knowledgebase.

Example:
	Question: Some issues happened, is there similar discussions?
	Answer: salesforce
	Question: Give me operational steps to resolve certain issue
	Answer: knowledgebase

Do not respond with the answer other than salesforce, knowledgebase.

Question: {query}
Answer:"""

class DSQuerier:
    def __init__(self, config):
        if CONFIG_BASIC_MODEL not in config:
            raise ValueError(f'The config doesn\'t contain {CONFIG_BASIC_MODEL}')
        self.basic_model = ModelManager(config[CONFIG_BASIC_MODEL])
        self.datasources = get_datasources(config)
        self.vector_store = VectorStore()

    def __judge_ds_type(self, query):
        prompt = PromptTemplate.from_template(CLASSIFICATION_PROMPT)
        chain = (
                {'query': RunnablePassthrough()}
                | prompt
                | self.basic_model.llm
                | StrOutputParser()
                )
        ds_type = chain.invoke(query)
        if ds_type not in self.datasources:
            return CONFIG_SF
        return ds_type

    def __get_ds(self, ds_type):
        if ds_type not in self.datasources:
            raise ValueError(f'Unknown datasource type: {ds_type}')
        return self.datasources[ds_type]

    def query(self, query, ds_type=None):
        if ds_type is None:
            ds_type = self.__judge_ds_type(query)
        ds = self.__get_ds(ds_type)
        docs = self.vector_store.similarity_search(ds_type,
                                                  ds.model_manager.embeddings,
                                                  query)
        return ds, docs[0]
