from typing import Union, Type

from langchain.schema import HumanMessage, AIMessage, FunctionMessage


LLM_Message = Type[Union[HumanMessage, AIMessage, FunctionMessage]]
