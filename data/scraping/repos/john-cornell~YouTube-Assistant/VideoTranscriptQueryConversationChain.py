from .chain_prompts import video_transcript_chat_prompt, video_transcript_prompt

from typing import List
from dotenv import load_dotenv
from chains.AgentTypeChain import AgentTypeChain
from chains.RagPromptOptimizerChain import RagPromptOptimizerChain
from embeddings_store import Embeddings
from vectorstorage.vectorstore import get_vectorstore_for_db
from multiagent.prompt_history_agent import prompt_history_agent

from typing import Any, Dict, List, Optional

from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.schema.language_model import BaseLanguageModel
from langchain.chains.base import Chain
from langchain.callbacks.manager import (
    CallbackManagerForChainRun
)

load_dotenv()

class VideoTranscriptQueryConversationChain(Chain):
    llm:BaseLanguageModel
    debug: bool
    k: int = 15
    embeddings: Embeddings = Embeddings.HUGGINGFACE
    agents : Dict = {}
    
    class Config:
        arbitrary_types_allowed = True
        extra = "forbid"

    @property
    def input_keys(self) -> List[str]:
        return ["url", "query"]

    @property
    def output_keys(self) -> List[str]:
        return ["output"]

    def _call(
            self,
            inputs: Dict[str, Any],
            run_manager: Optional[CallbackManagerForChainRun] = None
    ) -> Dict[str, str]:

        prompt_history_agent = self.get_history_agent()
        prompt = build_prompt()

        #Get inputs
        query = inputs["query"]        
        if not query:
            raise ValueError("query not set")

        url = inputs["url"]
        if not url:
            raise ValueError("url not set")


        #Assertain prompt type - only RAG is supported for now
        process_chain = AgentTypeChain(llm=self.llm, debug=True)
        process_response = process_chain.run({"query": query})["process"]

        print(f"Process response: {process_response}")

        #get history
        history = prompt_history_agent.format()

        if process_response == "SUMMARY":
            print("SUMMARY not yet supported")
            return self.build_return("That question would require a summarisation of the whole video, which is not yet supported.", [], history, None)
        if process_response == "NONE":
            
            prompt=build_chat_prompt()
            #Send simple prompt to LLM
            return self.get_response(prompt_history_agent, prompt, query, history, prompt_history_agent.format_docs())
        
        #Optimize RAG search prompt
        optimizer_chain = RagPromptOptimizerChain(llm=self.llm, debug=True)
        optimizer_response_json = optimizer_chain.run({"query": query})

        optimizer_response = optimizer_response_json["searchprompt"]

        #Get docs from db
        db = get_vectorstore_for_db(url, Embeddings.HUGGINGFACE)

        docs = db.similarity_search(optimizer_response, k=self.k)
        docs_page_content = [doc.page_content for doc in docs]

        prompt_history_agent.append_docs(docs_page_content)        

        #Send RAG prompt to LLM
        return self.get_response(prompt_history_agent, prompt, query, history, docs_page_content)

    def get_response(self, prompt_history_agent, prompt, query, history, docs_page_content):
        chain = LLMChain(llm=self.llm, prompt=prompt, output_key="answer")    

        response = chain.run(history=history, query=query, docs=docs_page_content)        
        response = response.replace("\r", "")

        prompt_history_agent.append_query(query=query)
        prompt_history_agent.append_response(response)

        return self.build_return(response, docs_page_content, history, prompt)

    def get_history_agent(self):
        if "agent" not in self.agents:
            self.agents["agent"] = prompt_history_agent(self.llm)
        
        return self.agents["agent"]

    def build_return(self, response, docs, history, prompt):
        return {
            "output":{
                "response": response,
                "metadata": {
                    "docs": docs,
                    "prompt": prompt,
                    "history": history
                }
            }
        }

def build_prompt():
    return PromptTemplate(
        input_variables=["history", "query", "docs"],
        template=video_transcript_prompt
    )

def build_chat_prompt():
    return PromptTemplate(
        input_variables=["history", "query", "docs"],
        template=video_transcript_chat_prompt
    )

