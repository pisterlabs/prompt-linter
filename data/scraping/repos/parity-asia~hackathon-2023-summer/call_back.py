# from typing import Dict, Union, Any, List

# from langchain.callbacks.base import AsyncCallbackHandler, BaseCallbackHandler
# from langchain.schema import AgentAction
# from langchain.agents import AgentType, initialize_agent, load_tools
# from langchain.callbacks import tracing_enabled
# from langchain.llms import OpenAI
# from services.helpers.question_db import ErrorQuestionRecord
# import json


# # First, define custom callback handler implementations
# class MyCustomHandler(AsyncCallbackHandler):
#     question = None
#     async def on_tool_start(
#         self, serialized: Dict[str, Any], input_str: str, **kwargs: Any
#     ) -> Any:
#         """Run when tool starts running."""
#         if serialized['name'] == 'Search something on internet':
#             question =  serialized['tool_input']

#     async def on_agent_finish(self, finish: AgentFinish, **kwargs: Any) -> Any:
#         print(f"on_agent_action {finish}")
#         await ErrorQuestionRecord.insert_error_data(question, json.dumps(finish.return_values), "can not find answer")

# # handler = MyCustomHandler()
# # result = agent.run(question, callbacks=[handler])
