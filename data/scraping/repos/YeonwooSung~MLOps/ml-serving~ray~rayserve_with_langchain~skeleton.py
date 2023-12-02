from ray import serve
from starlette.requests import Request
import os

from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]


@serve.deployment
class DeployLLM:
    def __init__(self):
        # We initialize the LLM, template and the chain here
        llm = OpenAI(openai_api_key=OPENAI_API_KEY)
        template = "Question: {question}\n\nAnswer: Let's think step by step."
        prompt = PromptTemplate(template=template, input_variables=["question"])
        self.chain = LLMChain(llm=llm, prompt=prompt)

    def _run_chain(self, text: str):
        return self.chain(text)

    async def __call__(self, request: Request):
        # 1. Parse the request
        text = request.query_params["text"]
        # 2. Run the chain
        resp = self._run_chain(text)
        # 3. Return the response
        return resp["text"]


# Bind the model to deployment
deployment = DeployLLM.bind()

# Example port number
PORT_NUMBER = 8282
# Run the deployment
serve.api.run(deployment, port=PORT_NUMBER)
