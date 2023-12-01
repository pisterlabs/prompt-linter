# Tools as OpenAI Functions
# Make sure langchain to 0.0.200
# pip install --upgrade --force-reinstall langchain

import json
from langchain.tools import MoveFileTool, format_tool_to_openai_function
from langchain.chat_models import ChatOpenAI

from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage
# from stock_tool import StockPriceTool
from poi import TravelPOITool

model = ChatOpenAI(model="gpt-3.5-turbo-0613")

tools = [TravelPOITool()]
functions = [format_tool_to_openai_function(t) for t in tools]
print(functions[0])

# tools = [StockPriceTool()]
# functions = [format_tool_to_openai_function(t) for t in tools]

# Prepare openai.functions
# tools = [StockPriceTool()]
# functions = [format_tool_to_openai_function(t) for t in tools]

while True:
    try:
        question = input("Question: ")
        hm = HumanMessage(content=question)
        ai_message = model.predict_messages([hm], functions=functions)
        ai_message.additional_kwargs['function_call']
        _args = json.loads(
            ai_message.additional_kwargs['function_call'].get('arguments'))
        tool_result = tools[0](_args)

        print("Answer: ", tool_result)
    except KeyboardInterrupt:
        break


# tools = [MoveFileTool()]
# functions = [format_tool_to_openai_function(t) for t in tools]

# message = model.predict_messages(
#     [HumanMessage(content='move file foo to bar')], functions=functions)

# print(message)

# print(message.additional_kwargs['function_call'])

# del_tools = [DeleteFileTool()]
# del_functions = [format_tool_to_openai_function(t) for t in del_tools]

# message = model.predict_messages(
#     [HumanMessage(content='delete *.md from Document folder')], functions=del_functions)

# print(message)

# print(message.additional_kwargs['function_call'])
