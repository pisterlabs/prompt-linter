import contextlib
import io
import re
from langchain.chat_models import ChatOpenAI
from langchain.prompts import SystemMessagePromptTemplate
from langchain.schema import AIMessage, HumanMessage, SystemMessage
from constant.prompt import action_prompt, action_response_prompt

# 导入所有的基础API
from service.order_service import *
from util.ansi_console_utils import print_action_agent_human_message


class ActionAgent:
    def __init__(
        self,
        model_name: str,
        request_timout: int,
        show_execution_error: bool,
    ):
        self.llm = ChatOpenAI(
            model_name=model_name,
            temperature=0,
            request_timeout=request_timout,
        )
        self.show_execution_error = show_execution_error

    def execute_code_and_gather_info(self, code: str) -> tuple[str, str]:
        error_message = ""
        captured_output = ""
        # Create a StringIO object
        output = io.StringIO()
        # Redirect standard output to the StringIO object
        with contextlib.redirect_stdout(output):
            try:
                exec(code)
            except Exception as e:
                error_message = str(e)
        # Get the contents of the StringIO object
        captured_output = output.getvalue()
        return error_message, captured_output

    def process_ai_message(self, ai_message: AIMessage) -> tuple[str, str]:
        program_call = ""
        program_code = ""
        code_pattern = re.compile(r"```(?:python)(.*?)```", re.DOTALL)
        code_match = code_pattern.search(ai_message.content)
        if code_match:
            program_code = code_match.group(1)
        else:
            raise ValueError("No program code block found")
        call_pattern = re.compile(r"```(?:call)(.*?)```", re.DOTALL)
        call_match = call_pattern.search(ai_message.content)
        if call_match:
            program_call = call_match.group(1)
            name_pattern = re.compile(r"(\w+)\(", re.DOTALL)
            name_match = re.search(name_pattern, program_call)
            if name_match:
                program_name = name_match.group(1)
            else:
                raise ValueError("No program name block found")
        else:
            raise ValueError("No program call block found")
        return program_call, program_code, program_name

    def render_human_message(
        self, code: str, error_message: str, output: str, critique: str, task: str
    ) -> HumanMessage:
        observation = ""
        if code:
            observation += f"Code: {code}\n\n"
        else:
            observation += f"Code: No code in the first round\n\n"
        if self.show_execution_error:
            if error_message:
                observation += f"Execution error: {error_message}\n\n"
            else:
                observation += f"Execution error: No error\n\n"
        if output:
            observation += f"Execution log: {output}\n\n"
        else:
            observation += f"Execution log: No output in the first round\n\n"
        if critique:
            observation += f"Critique: {critique}\n\n"
        else:
            observation += f"Critique: None\n\n"
        observation += f"Task: {task}\n\n"
        print_action_agent_human_message(message_content=observation)
        return HumanMessage(content=observation)

    def render_system_message(self, skills: list) -> SystemMessage:
        response_format = action_response_prompt.content
        system_message_template = SystemMessagePromptTemplate.from_template(
            template=action_prompt.content
        )
        system_message = system_message_template.format(
            skills=skills, response_format=response_format
        )
        assert isinstance(system_message, SystemMessage)
        return system_message
