import cohere
from ..llms.base import LLM
from typing import Any, Optional
from dotenv import load_dotenv
import logging
import sys
import os
import ast

logging.basicConfig(
    format='%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
    datefmt='%d-%b-%y %H:%M:%S',
    level=logging.INFO
)
logger = logging.getLogger('spiral.log')
load_dotenv()

def evaluate_ast(node):
    logger.info(node)
    if isinstance(node, ast.Num):
        return node.n
    elif isinstance(node, ast.BinOp):
        left = evaluate_ast(node.left)
        right = evaluate_ast(node.right)
        op = node.op
        if op == ast.Add:
            return left + right
        elif op == ast.Sub:
            return left - right
        elif op == ast.Mult:
            return left * right
        elif op == ast.Div:
            return left / right
        else:
            raise Exception('Unsupported operator: {}'.format(op))

calculator = {
    'name': 'Calculator',
    'description':
      """Useful for getting the result of a math expression.
      The input to this tool should be a valid mathematical expression that could be executed by a simple calculator.
      The code will be executed in a python environment so the input should be in a format it can be executed.
      
      Example:
      User: what is the square root of 25?
      action_input: 25**(1/2)""",
    'execute': evaluate_ast,
}

class Coral(LLM):
    """A class for interacting with the Coral API.

    Args:
        model: The name of the Coral model to use.
        temperature: The temperature to use when generating text.
        api_key: Your Coral API key.
    """
    model: str = 'command-nightly'
    temperature: float = 0.1
    chat_history: list[str] = []
    api_key: str = os.getenv('COHERE_API_KEY', '')

    def __call__(self, query, **kwds: Any)->str:
        """Generates a response to a query using the Coral API.

        Args:
        query: The query to generate a response to.
        kwds: Additional keyword arguments to pass to the Coral API.

        Returns:
        A string containing the generated response.
        """

        client = cohere.Client(api_key=self.api_key)
        response = client.chat( 
            model=self.model,
            message=query,
            temperature=self.temperature,
            prompt_truncation='auto',
            stream=False,
            citation_quality='accurate',
            connectors=[{"id": "web-search"}]
        )
            
        return response.text
    
if __name__ == "__main__":
    try:
        assistant = Coral()
        # assistant.add_tool(calculator)
        while True:
            message = input("\nEnter Query$ ")
            result = assistant(message)
            print(result)
    except KeyboardInterrupt:
        sys.exit(1)
