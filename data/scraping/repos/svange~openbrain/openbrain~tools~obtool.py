from langchain.tools import BaseTool
from openbrain.tools.protocols import OBCallbackHandlerFunctionProtocol


class OBTool:
    """A tool for GptAgents. Tools consist of the main langchain extended BaseTool and any callbacks needed to supplement"""

    on_llm_start: OBCallbackHandlerFunctionProtocol
    on_chat_model_start: OBCallbackHandlerFunctionProtocol
    on_llm_new_token: OBCallbackHandlerFunctionProtocol
    on_llm_end: OBCallbackHandlerFunctionProtocol
    on_llm_error: OBCallbackHandlerFunctionProtocol
    on_chain_start: OBCallbackHandlerFunctionProtocol
    on_chain_end: OBCallbackHandlerFunctionProtocol
    on_chain_error: OBCallbackHandlerFunctionProtocol
    on_tool_start: OBCallbackHandlerFunctionProtocol
    on_tool_end: OBCallbackHandlerFunctionProtocol
    on_tool_error: OBCallbackHandlerFunctionProtocol
    on_text: OBCallbackHandlerFunctionProtocol
    on_agent_action: OBCallbackHandlerFunctionProtocol
    on_agent_finish: OBCallbackHandlerFunctionProtocol
    tool: BaseTool
