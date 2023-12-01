from langchain.tools import Tool
from langchain.utilities import PythonREPL

python_repl = PythonREPL()

def get_tool():
    return [
        Tool(
            name="Python",
            func=python_repl.run,
            description="useful when you need to use logic in your answer. Input must be valid python code. You should always use print to output what you need to see."
            )
        ]