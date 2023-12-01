from langchain.tools import BaseTool

from typing import Optional, Type, Callable


class MultiInputTool(BaseTool):
    def __init__(self, name: str, func: Callable, num_inputs: int, input_parser: Optional[Callable] = None):
        super().__init__(name, func)
        self.num_inputs = num_inputs
        self.input_parser = input_parser if input_parser else lambda x: x