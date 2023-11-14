import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from langchain import PromptTemplate, OpenAI, LLMChain
import yaml


class RouterConfig:
    def __init__(self, llm=None, spec=None):
        self.chain_map = {}
        self.post_processor_map = {}
        self.pre_processor_map = {}
        chroma_client = chromadb.Client()
        sentence_transformer_ef = SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
        self.router_coll = chroma_client.create_collection(name='router', embedding_function=sentence_transformer_ef)
        if not llm:
            llm = OpenAI(temperature=0.9)
        with open(spec, 'r') as fp:
            content = yaml.safe_load(fp)
            for model in content.get('models'):
                for mname, mcontent in model.items():
                    mname = mname.lower()
                    self.router_coll.add(ids=[str(x) for x in range(len(mcontent.get('qa_maker')))],
                                         documents=mcontent.get('qa_maker'),
                                         metadatas=[{'classification': mname} for x in
                                                    range(len(mcontent.get('qa_maker')))])
                    self.chain_map[mname] = LLMChain(llm=llm, prompt=PromptTemplate(template=mcontent.get('template'),
                                                                                    input_variables=mcontent.get(
                                                                                        'input_vars')))
                    self.post_processor_map[mname] = mcontent.get('post_processor_script')
                    self.pre_processor_map[mname] = mcontent.get('pre_processor_script')

    def get_chains(self):
        return self.chain_map

    def get_embedding(self):
        return self.router_coll

    def get_post_processor_per_chain(self):
        return self.post_processor_map

    def get_pre_processor_per_chain(self):
        return self.pre_processor_map


if __name__ == '__main__':
    b = RouterConfig()
    c = b.get_chains()
