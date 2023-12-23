from langchain.tools import BaseTool
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from typing import  Dict, Optional
from langchain.pydantic_v1 import  Field
class ImagePathFinderTool(BaseTool):
    name = "Image Path Finder"
    description = "Use this tool only to find the exact path of an image if it's not in chat memory" \
                  "The input to this tool must be a comma seperated string of keys to search about"\
                  "It will return a simple string for the image path"
    paths = []
    def _run(self, query:str):

        prompt = PromptTemplate(
            template=
                    "Given the following paths"\
                    "{paths}"\
                    "Write only the path that most likely about {query}"
                    "If there is no path return N/A",
            input_variables=["query","paths"]
        )

        llm = OpenAI(temperature=0)
        # Run prompt through LLM here and return output
        llm_chain = LLMChain(llm=llm, prompt=prompt,output_key="path")
        result = llm_chain.run(query = query,paths=self.paths)
        if 'N/A' in result:
            self.return_direct = True
            return 'The image does not exist'
        #if result.strip() not in self.paths:
        #    return f"{result} Image Not Found"
        self.return_direct = False

        return result.replace("'","")

    def _arun(self, query: str):
        raise NotImplementedError("This tool does not support async")

