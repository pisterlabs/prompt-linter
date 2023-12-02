import time
import logging
import os
import shared.constants as constants
from documents.vector_database import get_database
from shared.selector import get_llm, get_embedding
from langchain.chains import RetrievalQA
import utilities.calculate_timing as calculate_timing
from ai.agent_tools.utilities.abstract_tool import AbstractTool

class VectorStoreRetrievalQATool(AbstractTool):

    def configure(self, database_name, run_locally, override_llm = None, top_k = 4, search_type = "mmr", chain_type = "stuff", search_distance = .5, verbose = False, max_tokens = constants.MAX_LOCAL_CONTEXT_SIZE, return_source_documents = False, return_direct = None):
            self.database_name = database_name
            self.top_k = top_k
            self.search_type = search_type
            self.search_distance = search_distance
            self.verbose = verbose
            self.max_tokens = max_tokens
            self.return_source_documents = return_source_documents

            # Load the specified database 
            self.db = get_database(get_embedding(run_locally), database_name)    

            vectordbkwargs = {"search_distance": search_distance, "k": top_k, "search_type": search_type}

            # Get the llm
            if override_llm != None:
                 llm = override_llm
            else:
                llm = get_llm(run_locally)
            
            self.retrieval_qa = RetrievalQA.from_chain_type(llm=llm, chain_type=chain_type, retriever=self.db.as_retriever(search_kwargs=vectordbkwargs), verbose=verbose, return_source_documents=return_source_documents)

            logging.debug(f"VectorStoreRetrievalQATool initialized with database_name={database_name}, top_k={top_k}, search_type={search_type}, search_distance={search_distance}, verbose={verbose}, max_tokens={max_tokens}")

    @property
    def database(self):
        return self.db

    def run(self, query:str) -> str:
        logging.debug(f"\nVectorStoreRetrievalQATool got: {query}")

        start_time = time.time()
        result = self.retrieval_qa({"query": query})
        end_time = time.time()
      
        elapsed_time = end_time - start_time
        logging.debug("Operation took: " + calculate_timing.convert_milliseconds_to_english(elapsed_time * 1000))

        result_string = result['result']

        # When this tool is used from the self_ask_agent_tool, it doesn't 
        if self.return_source_documents:
            # Append the source docs to the result, since we can't return anything else but a string??  langchain... more like lamechain, amirite??
            source_docs_list = [f"\t- {os.path.basename(d.metadata['source'])} page {int(d.metadata['page']) + 1}" if 'page' in d.metadata else f"\t- {os.path.basename(d.metadata['source'])}" for d in result["source_documents"]]
            unique_list = list(set(source_docs_list))
            unique_list.sort()
            source_docs = "\n".join(unique_list)
            result_string = result_string + "\nSource Documents:\n" + source_docs

        return result_string