import os
from dotenv import load_dotenv
from langchain.document_loaders import UnstructuredHTMLLoader
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI
from pydantic import BaseModel, validator, Field
from langchain.callbacks.manager import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
from langchain.tools import BaseTool
from typing import Optional, Type

# load dotenv
load_dotenv()
openai_api_key = os.getenv('OPENAI_API_KEY')

# You can provide a custom args schema to add descriptions or custom validation

llm = OpenAI(openai_api_key=openai_api_key,model_name='gpt-3.5-turbo-16k-0613',temperature=0)

context = """
    You are an intelligent html understanding agent. You will be given a html page,
    and a query about what to click on the page. You have to provide the 
    correct element selector string(ies) in list format
    """

example = """
    One example is: 

    input: <div class="filter-wrapper-platform">
	   <div class="plp-page-filters-v2 js-plp-filter-group pangea-inited" data-filter-id="platform">
	      <h2>Platform</h2>
	      <button class="toggle-visibility js-toggle-visibility" aria-label="Platform" aria-expanded="true"></button>
	      <div class="inner js-inner" style="display: block;">
	         <ul>
	            <li class="option">
                  <input name="platform-checkbox" type="checkbox" id="platform-checkbox-0" data-filter-group="platform" data-filter-val="windows" data-filter-title="Windows">
                  <label class="h6" for="platform-checkbox-0"><span class="filter-title">Windows</span>
                  </label>
	            </li>
	         
	            <li class="option">
                  <input name="platform-checkbox" type="checkbox" id="platform-checkbox-1" data-filter-group="platform" data-filter-val="mac" data-filter-title="Mac">
                  <label class="h6" for="platform-checkbox-1"><span class="filter-title">Mac</span>
                  </label>
	            </li>
	         
	            <li class="option">
                  <input name="platform-checkbox" type="checkbox" id="platform-checkbox-2" data-filter-group="platform" data-filter-val="linux" data-filter-title="Linux">
                  <label class="h6" for="platform-checkbox-2"><span class="filter-title">Linux</span>
                  </label>

    elements: ['input[data-filter-val="linux"]',
        'input[data-filter-val="mac"]']
        """ 

base_context = context + example

class WebActionSchema(BaseModel):
    query: str = Field(description="Name of the button or element to look on the html page along with any relevant additional information")
    html_code: str = Field(description="The html code of the page or html code chunk fetched by the agent to look for the element on")

class WebActionIdentifier(BaseTool):
    name = "web_action_identifier"
    description = "identify the correct element selector and action to take on a given web page"
    args_schema: Type[WebActionSchema] = WebActionSchema

    def _run(
        self, 
        query: str, 
        html_code: str,
        run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """Use the tool."""
        print("### web action identifier ###")
        return llm(base_context + "\n"+query + "\n" + html_code)

    async def _arun(
        self, 
        query: str,
        html_code: str, 
        # run_manager: Optional[AsyncCallbackManagerForToolRun] = None
    ) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError("custom_search does not support async")

#todo
# get agent user input
# get html
# define my input




