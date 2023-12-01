from typing import List
from openai.types.beta.threads.required_action_function_tool_call import RequiredActionFunctionToolCall
from openai.types.beta.threads.run_submit_tool_outputs_params import ToolOutput
import json
import functions

def execute(toolcall: RequiredActionFunctionToolCall) -> ToolOutput:
    function = toolcall.function
    selected_function = getattr(functions, function.name)
    arguments = json.loads(function.arguments)
    
    if selected_function is not None and callable(selected_function):
        return {"tool_call_id": toolcall.id, "output": json.dumps(selected_function(**arguments))}
    
def execute_all(toolcalls: List[RequiredActionFunctionToolCall]) -> List[ToolOutput]:
    result: List[ToolOutput] = []
    for toolcall in toolcalls:
        result.append(execute(toolcall))
        
    return result