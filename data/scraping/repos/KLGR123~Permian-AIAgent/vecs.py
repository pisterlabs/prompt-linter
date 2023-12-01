from langchain.agents import load_tools, Tool, tool
from langchain.agents import initialize_agent
from langchain.llms import OpenAI, OpenAIChat
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Qdrant, Chroma, Pinecone, FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import TextLoader
from langchain.chains import RetrievalQA
from llama_index import GPTSimpleVectorIndex, SimpleDirectoryReader, Document
from llama_index.readers.qdrant import QdrantReader
from llama_index.optimization.optimizer import SentenceEmbeddingOptimizer

from gptcache import cache
from gptcache.adapter import openai
import pinecone, warnings, yaml, os
warnings.filterwarnings("ignore")


class KapwingVectorStore:
    def __init__(self, chunk_size=10000,
                       model_name="gpt-4",
                       temperature=0,
                       filepath='data/query_instructions.txt'):

        cache.init()
        cache.set_openai_key()
        
        with open("config/config.yaml", 'r') as stream:
            config = yaml.safe_load(stream)

        os.environ["OPENAI_API_KEY"] = config['OPENAI_API_KEY']
        self.qdrant_host = config['QDRANT_HOST']
        self.qdrant_api_key = config['QDRANT_API_KEY']
        self.pcone_api_key = config['PINECONE_API_KEY']
        self.pcone_env = config['PINECONE_HOST']

        self.text_splitter = CharacterTextSplitter(separator="\n\n\n", chunk_size=chunk_size, chunk_overlap=0)
        self.loader = TextLoader(filepath)
        self.docs = self.text_splitter.split_documents(self.loader.load())

        self.embeddings = OpenAIEmbeddings()
        self.llm_ = OpenAIChat(model_name=model_name, temperature=temperature)
        
        with open("config/prompts.yaml", 'r') as stream:
            prompts = yaml.safe_load(stream)

        self.prefix = prompts['vectorstore_prefix']
        self.suffix = prompts['vectorstore_suffix']
        self.simple_scripts_prompt = prompts['simple_scripts_prompt']
        self.mask_prompt = prompts['mask_prompt']
        self.fast_func_prompt = prompts['fast_func_prompt']

    def get_qdrant(self):
        self.qdrant_tool_db = Qdrant.from_documents(self.docs, self.embeddings, host=self.qdrant_host, prefer_grpc=True, api_key=self.qdrant_api_key).as_retriever()
        self.qdrant_tool_vec = RetrievalQA.from_llm(llm=self.llm_, retriever=self.qdrant_tool_db)
        return self.qdrant_tool_db, self.qdrant_tool_vec

    def get_faiss(self):
        self.faiss_tool_db = FAISS.from_documents(self.docs, self.embeddings).as_retriever()
        self.faiss_tool_vec = RetrievalQA.from_llm(llm=self.llm_, retriever=self.faiss_tool_db)
        return self.faiss_tool_db, self.faiss_tool_vec

    def get_chroma(self):
        self.chroma_tool_db = Chroma.from_documents(self.docs, self.embeddings, collection_name="tools").as_retriever()
        self.chroma_tool_vec = RetrievalQA.from_llm(llm=self.llm_, retriever=self.chroma_tool_db)
        return self.chroma_tool_db, self.chroma_tool_vec

    def get_pcone(self):
        pinecone.init(api_key=self.pcone_api_key, environment=self.pcone_env)
        self.pcone_tool_db = Pinecone.from_documents(self.docs, self.embeddings, index_name="tool-db").as_retriever()
        self.pcone_tool_vec = RetrievalQA.from_llm(llm=self.llm_, retriever=self.pcone_tool_db)
        return self.pcone_tool_db, self.pcone_tool_vec

    def set_gpt_index(self):
        self.gpt_docs = [Document(doc.page_content) for doc in self.docs]
        self.tool_index = GPTSimpleVectorIndex.from_documents(self.gpt_docs)

    def gpt_index_query(self, query):
        res = self.tool_index.query(self.prefix.format(query=query) + self.mask_prompt + self.suffix,
                                    similarity_top_k=3
                                    # optimizer=SentenceEmbeddingOptimizer(percentile_cutoff=0.3)
                                )
        return res
        
    def gpt_index_funcs(self, query):
        res = self.tool_index.query(self.fast_func_prompt.format(query=query),
                                    similarity_top_k=3
                                    # optimizer=SentenceEmbeddingOptimizer(percentile_cutoff=0.3)
                                )
        return res

    def gpt_index_scripts_query(self, query):
        res = self.tool_index.query(self.simple_scripts_prompt.format(query=query) + self.mask_prompt, 
                                    # similarity_top_k=3,
                                    # optimizer=SentenceEmbeddingOptimizer(percentile_cutoff=0.3)
                                )
        return res

    def qdrant_query(self, query):
        res = self.qdrant_tool_vec.run(self.prefix.format(query=query) + self.suffix)
        return res

    def pcone_query(self, query):
        res = self.pcone_tool_vec.run(self.prefix.format(query=query) + self.suffix)
        return res

    def faiss_query(self, query):
        res = self.faiss_tool_vec.run(self.prefix.format(query=query) + self.suffix)
        return res
    
    def faiss_scripts_query(self, query):
        res = self.faiss_tool_vec.run(self.simple_scripts_prompt.format(query=query) + self.mask_prompt)
        return res


def main():
    query = input("QUERY: ")
    vec = KapwingVectorStore()
    vec.get_faiss()
    res = vec.faiss_query(query)
    print(res)

if __name__ == "__main__":
    main()
        
