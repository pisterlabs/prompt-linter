from langchain.chat_models import ChatOpenAI
from langchain.prompts import SystemMessagePromptTemplate
from langchain.schema import AIMessage, HumanMessage, SystemMessage
from prompts import load_prompt
import utils as U
from control_primitives_context import load_control_primitives_context
import regex as re
import time
import anthropic
import os


resources = "/opt/factorio/script-output/resource_data.json"
RESTRICTED_KEYWORDS = ['script.on_event', 'raise_event']


class ActionAgent:
    def __init__(
        self,
        env,
        model_name="claude-instant-v1",
        temperature=0,
        request_timout=120,
        ckpt_dir="ckpt",
        resume=False,
        chat_log=True,
        execution_error=True,
    ):
        self.env = env
        self.ckpt_dir = ckpt_dir
        self.chat_log = chat_log
        self.execution_error = execution_error
        U.f_mkdir(f"{ckpt_dir}/action")

        self.llm = anthropic.Client(os.environ["ANTHROPIC_API_KEY"])

    """
        Save entity information to /action/entity_memory.json
    """

    def update_entity_memory(self):
        pass

    """
        Save inventory information to /action/inventory_memory.json
    """

    def update_inventory_memory(self):
        pass

    def update_resource_memory(self):
        """
        Save resource information to /action/resource_memory.json
        """
        deposits = U.mod_utils.resource_clustering()
        U.dump_json(deposits, f"{self.ckpt_dir}/action/resources.json")

    def render_system_message(self, skills=[]):
        system_template = load_prompt("action_template")
        # FIXME: Hardcoded control_primitives
        base_skills = [
            "move",
            "craft",
            "build",
        ]
 
        programs = "\n\n".join(load_control_primitives_context(base_skills) + skills)
        response_format = load_prompt("action_response_format")
        system_message_prompt = SystemMessagePromptTemplate.from_template(
            system_template
        )
        system_message = system_message_prompt.format(
            programs=programs, response_format=response_format
        )
        assert isinstance(system_message, SystemMessage)
        print(f"System Message: {system_message}")
        return system_message

    def process_ai_message(self, message):
        error = None
        code_pattern = re.compile(r"Lua code:\n```(.*?)```", re.DOTALL)
        function_pattern = re.compile(r'Function name: (.*?)', re.DOTALL)

        code = code_pattern.findall(message['completion'])
        function = function_pattern.findall(message['completion'])
        assert(len(code) > 0),"Code not found"
        assert(len(function) > 0 ),"Function name not found"

        # assert(U.mod_utils.lua_code_verifier(code[0], RESTRICTED_KEYWORDS)), "Invalid lua code"


        return({
            "program_code": code[0],
            "function_name": function[0],
        })
    
    def render_human_message(
        self, *, events, code="", task="", context="", critique=""
    ):
        chat_messages = []
        error_messages = []
        
        observation = ""

        if code:
            observation += f"Code from the last round:\n{code}\n\n"
        else:
            observation += f"Code from the last round: No code in the first round\n\n"

        if self.execution_error:
            if error_messages:
                error = "\n".join(error_messages)
                observation += f"Execution error:\n{error}\n\n"
            else:
                observation += f"Execution error: No error\n\n"

        if self.chat_log:
            if chat_messages:
                chat_log = "\n".join(chat_messages)
                observation += f"Chat log: {chat_log}\n\n"
            else:
                observation += f"Chat log: None\n\n"

        # observation += f"Health: {health:.1f}/20\n\n"

        observation += f"Entity observations around the player:\n"

        #events are in the format
        #[{name:data}]
        for event in events:
            for name,data in event.items():
                observation += f"{name} - {data}\n"

        # observation += f"Position: x={position['x']:.1f}, y={position['y']:.1f}\n\n"

        observation += f"Task: {task}\n\n"

        if context:
            observation += f"Context: {context}\n\n"
        else:
            observation += f"Context: None\n\n"

        if critique:
            observation += f"Critique: {critique}\n\n"
        else:
            observation += f"Critique: None\n\n"

        return HumanMessage(content=observation)
