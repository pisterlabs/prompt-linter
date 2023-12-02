from langchain.tools import AIPluginTool, OpenAPISpec, BaseTool
from langchain.chat_models import ChatOpenAI
from langchain.agents import Tool
from urllib.parse import urlparse
import json

PROMPT = """You are a request creation tool for {plugin_name}. You will be given an API spec list along with a description of what goal should be achieved via the API (in natural language). Your job is to build a set of request parameters that will accomplish the set goal.
When building a request, you must always give your output in the following format: 
---
{
    "method": "GET",
    "path": "/openai_endpoint?example_param=shampoo"
}
---
{
    "method": "POST",
    "path": "/post_endpoint",
    "json": {
        "query": "shampoo",
        "filter": {
            "price_max": 12
        }
    }
}
---
You will never suggest a path that was not found within the API spec. You always achieve the request goal using the resources from the API spec.
Begin!
Here is the API spec: 
{api_spec}

Build a set of request parmeters for the following goal: {request}
---
{
    "method": "
"""

class _GetParamTool(BaseTool):
    name = 'Get Request Params'
    description=''
    func=print

    def _run(self, request: str):
        return self.func(request)
    async def _arun(self, request: str):
        return self.func(request)
    

class PluginLoader(object):
    def __init__(self, plugin_url: str, prompt=PROMPT):
        self.plugin = AIPluginTool.from_plugin_url(plugin_url)
        self.llm = ChatOpenAI(temperature=0, model_name='gpt-4')
        self.prompt = prompt.replace('{api_spec}', self.plugin.api_spec).replace('{plugin_name}', self.plugin.plugin.name_for_human) # not using format as it breaks the prompt

        self.api_base = self._get_api_base(self.plugin, plugin_url)

    def _get_api_base(self, plugin: AIPluginTool, url: str):
        specs = OpenAPISpec.from_url(plugin.plugin.api.url)
        server = specs.servers[0].url
        if server == '/':
            _ = urlparse(url)
            return f'{_.scheme}://{_.netloc}/'
        else:
            return server

    def build(self, request: str) -> str:
        resp = self.llm.call_as_llm(self.prompt.replace('{request}', request))
        try:
            resp = json.loads('{\n    "method": "' + resp)
            resp['path'] = self.api_base + resp['path']
        except json.JSONDecodeError:
            pass
        finally:
            return resp

    def get_tool(self):
        tool = _GetParamTool()
        tool.func = self.build
        tool.description = (
            f"Gets request parameters for interacting with {self.plugin.plugin.name_for_human} / {self.plugin.name}."
            f"Useful for {self.plugin.plugin.description_for_human}"
            "Accepts one argument, which should be the purpose of the request in natural language (e.g., `get products with 'shampoo' in the name`)."
            "Returns a string containing the type of request to preform, along with the URL, and (if it's a POST request) the json data to send. The result of this tool should be used with a request type tool."
        )
        return tool
        return Tool(
            name='Get Request Params',
            func=self.build,
            description=(
                f"Gets request parameters for interacting with {self.plugin.plugin.name_for_human} / {self.plugin.name}."
                f"Useful for {self.plugin.plugin.description_for_human}"
                "Accepts one argument, which should be the purpose of the request in natural language (e.g., `get products with 'shampoo' in the name`)."
                "Returns a string containing the type of request to preform, along with the URL, and (if it's a POST request) the json data to send. The result of this tool should be used with a request type tool."
            )
        )