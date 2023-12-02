from uuid import UUID
from langchain.agents import Tool, AgentExecutor, LLMSingleActionAgent, AgentOutputParser, initialize_agent
from langchain.prompts import HumanMessagePromptTemplate, SystemMessagePromptTemplate, ChatPromptTemplate, AIMessagePromptTemplate, PromptTemplate
from langchain import OpenAI, SerpAPIWrapper, LLMChain
from langchain.memory import ConversationBufferMemory, CombinedMemory
from langchain.chat_models import ChatOpenAI
from langchain.schema import AgentAction, AgentFinish, OutputParserException
from typing import Any, Dict, List, Optional, Union
import re
from langchain.schema.agent import AgentAction
from langchain.schema.messages import BaseMessage
from role_playing_system_prompt import role_playing_system_prompt
from role_playing_human_prompt import role_playing_human_prompt
from ..common.tools import notepad, relative_date_calculator, send_text_message, order_notepad
from ..knowledge_base.kb import knowledge_base
from toolkit_reservation_manager import ReservationsToolkit, CasualDiningReservationsToolkit
import langchain
import ast
from langchain.tools import tool
from pydantic import BaseModel
from langchain.callbacks import StdOutCallbackHandler
from langchain.callbacks.base import BaseCallbackHandler, BaseCallbackManager
from langchain.callbacks.manager import Callbacks
from langchain.utils.input import print_text
from ..common.utils import SCRATCH_SPACE_DIR_NAME, SCRATCH_SPACE_DIR_PATH
import json
import os

role_playing_system_message_prompt = SystemMessagePromptTemplate(prompt=role_playing_system_prompt)
role_playing_human_message_prompt = HumanMessagePromptTemplate(prompt=role_playing_human_prompt)

STOP_TOKENS = ["\nfunction_return:"]

langchain.verbose = True

AGENT_DIR_PREFIX = "role_playing"
AGENT_DIR_PATH = f"{SCRATCH_SPACE_DIR_PATH}/{AGENT_DIR_PREFIX}"
os.mkdir(AGENT_DIR_PATH)

system_prompt_file = open(f"{AGENT_DIR_PATH}/system_prompt.txt", "w")
system_prompt_file.write(role_playing_system_prompt.format())

_file_execution = None
_count = 0

class ToolCallbackHandler(BaseCallbackHandler):
    global _file_execution

    def on_tool_end(self, output: str, *, run_id: UUID, parent_run_id: UUID | None = None, **kwargs: Any) -> Any:
        result = super().on_tool_end(output, run_id=run_id, parent_run_id=parent_run_id, **kwargs)
        #print(f"ToolCallbackHandler: on_tool_end output = {output}")
        _file_execution.write(f"\nfunction_return: {output}\nfunction_return_extraction: ")
        return result
    
    def on_chain_start(self, serialized: Dict[str, Any], inputs: Dict[str   , Any], *, run_id: UUID, parent_run_id: UUID | None = None, tags: List[str] | None = None, metadata: Dict[str, Any] | None = None, **kwargs: Any) -> Any:
        result = super().on_chain_start(serialized, inputs, run_id=run_id, parent_run_id=parent_run_id, tags=tags, metadata=metadata, **kwargs)
        #print(f"ToolCallbackHandler: on_chain_start inputs = {inputs}")
        #if 'intermediate_steps' in inputs and len(inputs['intermediate_steps']) == 0:
        #    print(f"on_chain_start = serialized = {json.dumps(serialized, indent=4)}")
        #    print("---------- Prompt")
        #    print(serialized["kwargs"]["prompt"])
        #   print("---------- Formatted Prompt")
        #    print(serialized["kwargs"]["prompt"].format())
        return result
    
    def on_chain_end(self, outputs: Dict[str, Any], *, run_id: UUID, parent_run_id: UUID | None = None, **kwargs: Any) -> Any:
        result = super().on_chain_end(outputs, run_id=run_id, parent_run_id=parent_run_id, **kwargs)
        if 'text' in outputs: # Not logging the 'output' once the agent has finished
            _file_execution.write(outputs['text'])
        return result

class RolePlayingAgentOutputParser(AgentOutputParser):
    def parse(self, llm_output: str) -> Union[AgentAction, AgentFinish]:
        #print("---------------------------------- RolePlayingAgentOutputParser ----------------------------------")
        #print(llm_output)
        #print("Done with llm_output")
        if "[end]" in llm_output or "final_answer:" in llm_output or "reasoned_answer:" in llm_output:
            #print("Inside end")
            final_answer_match = re.search(r"final_answer:\s*(.*?)(?=\n|$)", llm_output)
            final_answer = final_answer_match.group(1) if final_answer_match else None

            if final_answer is None: # Most probably the agent stopped at "reasoned_answer"
                final_answer_match = re.search(r"reasoned_answer:\s*(.*?)(?=\n|$)", llm_output)
                final_answer = final_answer_match.group(1) if final_answer_match else None

            if final_answer is None: 
                raise ValueError("final_answer is None")
            
            final_answer = final_answer.replace("[end]", "")

            is_part_of_conversation  = bool(re.search(r"\[JMP\]:.*?\"json_formatted_function_input\"(?=\n|$)", llm_output))
            #print(f"is_part_of_conversation = {is_part_of_conversation}")
            #print(f"parsed_output = {final_answer}")
            #print("Done with end")
            #print("_______________________________________________________________________________________________")
            return AgentFinish(
                # Return values is generally always a dictionary with a single `output` key
                # It is not recommended to try anything else at the moment :)
                return_values={"output":final_answer},
                log=llm_output,
            )        

        # Parse out the Function and Function input
        thought_match = re.search(r"thought:\s*(.*?)(?=\n|$)", llm_output)
        function_name_match = re.search(r"function_name:\s*(.*?)(?=\n|$)", llm_output)
        function_input_arguments_value_pairs_match = re.search(r"function_input_arguments_value_pairs:\s*(.*?)(?=\n|$)", llm_output)
        is_any_argument_value_missing_match = re.search(r"is_any_argument_value_missing:\s*(.*?)(?=\n|$)", llm_output)
        jmp_match = re.search(r"\[JMP\]:\s*(.*?)(?=\n|$)", llm_output)
        json_formatted_function_input_match = re.search(r"json_formatted_function_input:\s*(.*?)(?=\n|$)", llm_output)

        thought = thought_match.group(1) if thought_match else None
        function_name = function_name_match.group(1) if function_name_match else None
        function_input_arguments_value_pairs = function_input_arguments_value_pairs_match.group(1) if function_input_arguments_value_pairs_match else None
        is_any_argument_value_missing = is_any_argument_value_missing_match.group(1) if is_any_argument_value_missing_match else None
        jmp = jmp_match.group(1) if jmp_match else None
        json_formatted_function_input = json_formatted_function_input_match.group(1) if json_formatted_function_input_match else None

        #print("thought: ", thought)
        #print("function_name: ", function_name)
        #print("function_input_arguments_value_pairs: ", function_input_arguments_value_pairs)
        #print("is_any_argument_value_missing: ", is_any_argument_value_missing)
        #print("jmp: ", jmp)
        #print("json_formatted_function_input: ", json_formatted_function_input)

        # Extract the argument
        print(f"arguments = {json_formatted_function_input}")
        arg_str = json_formatted_function_input.strip() 
        print("crashed here")

        # Type cast the argument
        typed_arg: Union[str, dict] = None
        if arg_str:
            try:
                typed_arg = ast.literal_eval(arg_str)
            except (SyntaxError, ValueError):
                typed_arg = arg_str  # If evaluation fails, retain the original string representation

        if typed_arg is None:
            typed_arg = ""

        #print("___________________________ Calling Function _____________________________________")
        # Return the action and action input
        return AgentAction(tool=function_name, tool_input=typed_arg, log=llm_output)


output_parser = RolePlayingAgentOutputParser()

history = [role_playing_system_message_prompt, role_playing_human_message_prompt]
#llm = ChatOpenAI(temperature=0, model="gpt-4")
llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo")
chat_prompt = ChatPromptTemplate.from_messages(history)
conversation_buffer_memory = ConversationBufferMemory()
llm_chain = LLMChain(
    llm=llm, 
    prompt=chat_prompt
)
tools = [order_notepad, relative_date_calculator] + CasualDiningReservationsToolkit().get_tools()
tool_names = [tool.name for tool in tools]

role_playing_agent = LLMSingleActionAgent(
    llm_chain=llm_chain,
    output_parser=output_parser,
    stop=STOP_TOKENS,
    allowed_tools=tool_names,
)

role_playing_agent_executor = AgentExecutor.from_agent_and_tools(
    agent=role_playing_agent, 
    tools=tools,
    memory=conversation_buffer_memory
)

class RolePlayingZeroShotAgentSchema(BaseModel):
    question: str

@tool("assistant", args_schema=RolePlayingZeroShotAgentSchema)
def assistant(question: str):
    """ Assistant """
    global _file_execution, _count
    _file_execution = open(f"{AGENT_DIR_PATH}/execution_{_count}.txt", "w")
    _file_execution.write(f"[start]\nquestion: {question}\n")
    _count = _count + 1
    return role_playing_agent_executor.run(
        input = question,
        callbacks=[ToolCallbackHandler()]
    )

def direct_call_assistant(question: str):
    """ Assistant """
    global _file_execution, _count
    _file_execution = open(f"{AGENT_DIR_PATH}/direct_call_execution_{_count}.txt", "w")
    _file_execution.write(f"[start]\nquestion: {question}\n")
    _count = _count + 1
    return role_playing_agent_executor.run(
        input = question,
        callbacks=[ToolCallbackHandler()]
    )

