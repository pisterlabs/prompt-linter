from langchain.tools import BaseTool
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from typing import  Dict, Optional
from langchain.pydantic_v1 import  Field
class FilePathFinderTool(BaseTool):
    name = "File Path Finder"
    description = "Use this tool only to find the exact path of a file if it's not in chat memory" \
                  "The input to this tool must be a comma seperated string of keys to search about"\
                  "It will return a simple string for the image path"
    paths = []
    #paths: Optional[Dict] = Field(default_factory=dict)
    def _run(self, query:str):
        self.return_direct = False
        prompt = PromptTemplate(
            template=
                    "Given the following paths"\
                    "{paths}"\
                    "Write only the full file path as it is in the paths that most likely about {query}"
                    "If there is no path return N/A",
            input_variables=["query","paths"]
        )

        llm = OpenAI(temperature=0)
        # Run prompt through LLM here and return output
        llm_chain = LLMChain(llm=llm, prompt=prompt,output_key="path")
        result = llm_chain.run(query = query,paths=self.paths)
        if 'N/A' in result:
            self.return_direct = True
            return 'The file does not exist'
        #if result.strip() not in self.paths:
        #    return f"{result} Image Not Found"
        return result.replace("'","")

    def _arun(self, query: str):
        raise NotImplementedError("This tool does not support async")

