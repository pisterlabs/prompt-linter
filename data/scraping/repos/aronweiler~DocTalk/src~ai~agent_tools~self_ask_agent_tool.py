import time
import logging
import shared.constants as constants
from shared.selector import get_llm
from langchain.agents import initialize_agent
from langchain.agents import AgentType
from langchain.agents import  Tool
import utilities.calculate_timing as calculate_timing
from ai.agent_tools.utilities.abstract_tool import AbstractTool

class SelfAskAgentTool(AbstractTool):

    def configure(self, memory, local, search_tool:Tool, verbose = False, max_tokens = constants.MAX_LOCAL_CONTEXT_SIZE, override_llm = None):            
            self.verbose = verbose
            self.max_tokens = max_tokens
            self.search_tool = search_tool

            if(override_llm == None):
                llm = get_llm(local=local)
            else:
                llm = override_llm
            
            self.agent_chain = initialize_agent([search_tool], llm, agent=AgentType.SELF_ASK_WITH_SEARCH, verbose=verbose, memory=memory)

            logging.debug(f"SelfAskAgentTool initialized with local={local}, search_tool={search_tool}, verbose={verbose}, max_tokens={max_tokens}")

    def run(self, query:str) -> str:        
        logging.debug(f"\nSelfAskAgentTool got: {query}")

        start_time = time.time()
        result = self.agent_chain.run(input=query)
        end_time = time.time()
      
        elapsed_time = end_time - start_time
        logging.debug("Operation took: " + calculate_timing.convert_milliseconds_to_english(elapsed_time * 1000))

        ## If only I could return document sources with self-ask, that would be great... 
        # result_string = result['result']

        # if self.return_source_documents:
        #     # Append the source docs to the result, since we can't return anything else but a string??  langchain... more like lamechain, amirite??
        #     source_docs_list = [f"\t- {os.path.basename(d.metadata['source'])} page {int(d.metadata['page']) + 1}" if 'page' in d.metadata else f"\t- {os.path.basename(d.metadata['source'])}" for d in result["source_documents"]]
        #     unique_list = list(set(source_docs_list))
        #     unique_list.sort()
        #     source_docs = "\n".join(unique_list)
        #     result_string = result_string + "\nSource Documents:\n" + source_docs

        return result