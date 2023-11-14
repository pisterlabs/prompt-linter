from typing import Optional
from langchain.tools import BaseTool
from langchain.chains.summarize import load_summarize_chain
from langchain.chat_models import ChatOpenAI
from langchain.vectorstores import FAISS

from langchain import PromptTemplate
from utils.callback import CustomHandler


class SummarizeFileTool(BaseTool):
    name = "Summarize File"

    description = "Use Path Finder Tool before using this tool if the file path is not in memory" \
                  "Use this tool to summarize the content of a file" \
                  "The input to this tool should be a comma separated list of strings of length two, representing the file path and what to search for(can be empty string if not given)."
    vectorstore:FAISS
    paths = []

    def _run(self,input):
        file_path,prompt = input.split(",")

        file_path_exists = False
        for item in self.paths:
            if item.get("file_path") == file_path:
                file_path_exists = True
                break
        if(not file_path_exists):
            self.return_direct = True
            return "Please provide the file name"

        llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo-16k",streaming=True)
        prompt_template = """

        {text}

        SUMMARY:"""
        PROMPT = PromptTemplate(template=prompt_template, input_variables=["text"])

        chain = load_summarize_chain(llm, chain_type="map_reduce", 
                                    map_prompt=PROMPT, combine_prompt=PROMPT)
        
  
        if(prompt != ''):
            retriever = self.vectorstore.as_retriever(
            
            search_kwargs={'k':100000,'filter': {'source':file_path.replace('/','\\')}}
            )
            all_documents = retriever.get_relevant_documents(prompt)
        else:
            all_documents = list(self.vectorstore.docstore._dict.values())


        if(len(all_documents) == 0):
            self.return_direct = True
            return "No information found"
        self.return_direct = True
        return chain.run(all_documents)
    def _arun(self, query: str):
        raise NotImplementedError("This tool does not support async")

