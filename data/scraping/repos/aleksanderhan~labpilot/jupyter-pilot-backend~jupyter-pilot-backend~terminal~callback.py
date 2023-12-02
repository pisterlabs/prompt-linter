import json
from langchain.callbacks.base import BaseCallbackHandler, AsyncCallbackHandler
from typing import List, Dict, Any, Union
from langchain.schema import LLMResult, AgentAction


class DefaultCallbackHandler(AsyncCallbackHandler):
    def __init__(self, ws):
        self.terminal_ws = ws

    async def on_llm_new_token(self, token: str, **kwargs) -> None:
        if token:
            reply = {
                "message": token, 
                "done": False,
                "start": False,
                "method": "default"
            }
            await self.terminal_ws.send(json.dumps(reply))

    async def on_llm_start(self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any) -> None:
        reply = {
            "done": False,
            "start": True,
            "method": "default"
        }
        await self.terminal_ws.send(json.dumps(reply))

    async def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        reply = {
            "done": True,
            "start": False,
            "method": "default"
        }
        await self.terminal_ws.send(json.dumps(reply))
    
    async def on_llm_error(self, error: Union[Exception, KeyboardInterrupt], **kwargs: Any) -> Any:
        print("DefaultCallbackHandler.error error: ", error)
        print("DefaultCallbackHandler.error kwargs: ", error)

    async def on_chain_start(self, serialized: Dict[str, Any], inputs: Dict[str, Any], **kwargs: Any) -> Any:
        print(f"DefaultCallbackHandler.on_chain_start serialized: {serialized}")
        print(f"DefaultCallbackHandler.on_chain_start inputs: {inputs}")
        print(f"DefaultCallbackHandler.on_chain_start kwargs: {kwargs}")


    async def on_tool_start(self, serialized: Dict[str, Any], input_str: str, **kwargs: Any) -> Any:
        print(f"DefaultCallbackHandler.on_tool_start serialized: {serialized}")
        print(f"DefaultCallbackHandler.on_tool_start input_str: {input_str}")
        print(f"DefaultCallbackHandler.on_tool_start kwargs: {kwargs}")

    async def on_agent_action(self, action: AgentAction, **kwargs: Any) -> Any:
        print(f"DefaultCallbackHandler.on_agent_action action: {action}")
        print(f"DefaultCallbackHandler.on_agent_action kwargs: {kwargs}")




class PrintCallbackHandler(AsyncCallbackHandler):

    async def on_llm_new_token(self, token: str, **kwargs) -> None:
        if token:
            reply = {
                "message": token, 
                "done": False,
                "start": False,
                "method": "default"
            }
            print("PrintCallbackHandler.on_llm_new_token reply", reply)

    async def on_llm_start(self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any) -> None:
        reply = {
            "done": False,
            "start": True,
            "method": "default"
        }
        print("PrintCallbackHandler.on_llm_start reply", reply)

    async def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        reply = {
            "done": True,
            "start": False,
            "method": "default"
        }
        print("PrintCallbackHandler.on_llm_end reply", reply)
    
    async def on_llm_error(self, error: Union[Exception, KeyboardInterrupt], **kwargs: Any) -> Any:
        print("PrintCallbackHandler.error", error)

    async def on_chain_start(self, serialized: Dict[str, Any], inputs: Dict[str, Any], **kwargs: Any) -> Any:
        print(f"PrintCallbackHandler.on_chain_start serialized: {serialized} \ninputs: {inputs}")

    async def on_tool_start(self, serialized: Dict[str, Any], input_str: str, **kwargs: Any) -> Any:
        print(f"PrintCallbackHandler.on_tool_start serialized: {serialized} \ninput_str: {input_str}")

    async def on_agent_action(self, action: AgentAction, **kwargs: Any) -> Any:
        print(f"PrintCallbackHandler.on_agent_action {action}")
