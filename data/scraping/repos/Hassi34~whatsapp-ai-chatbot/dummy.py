
import boto3
from dotenv import load_dotenv
from langchain.tools import BaseTool

load_dotenv()

ec2_client = boto3.client('ec2')

desc = """Don't use this tool in anycase
          It doesn't provide any functionality
"""

class DummyTool(BaseTool):
    name = "Dummy Tool"
    description = desc
    def _run(self):...
    
    def _arun(self, query: str):
        raise NotImplementedError("This tool does not support async")
