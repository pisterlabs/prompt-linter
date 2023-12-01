from flaml.autogen.agent import ResponsiveAgent
from collections import defaultdict
import gym
import random
import json
import time
from typing import Callable, Dict, Optional, Union
from typing import Any, Callable, Dict, List, Optional, Union
import openai
from selenium.webdriver.common.keys import Keys
import json
import os
from computergym.miniwob.miniwob_interface.action import (
    MiniWoBType,
    MiniWoBElementClickId,
    MiniWoBElementClickXpath,
    MiniWoBElementClickOption,
    MiniWoBMoveXpath,
)
import re
import matplotlib.pyplot as plt

def last_boxed_only_string(string: str) -> Optional[str]:
    """Source: https://github.com/hendrycks/math
    Extract the last \\boxed{...} or \\fbox{...} element from a string.
    """
    if "\\boxed" in string:
        idx = string.rfind("\\boxed")
        if idx < 0:
            idx = string.rfind("\\fbox")
            if idx < 0:
                return None

        i = idx
        right_brace_idx = None
        num_left_braces_open = 0
        while i < len(string):
            if string[i] == "{":
                num_left_braces_open += 1
            if string[i] == "}":
                num_left_braces_open -= 1
                if num_left_braces_open == 0:
                    right_brace_idx = i
                    break
            i += 1

        if right_brace_idx is None:
            retval = None
        else:
            retval = string[idx : right_brace_idx + 1]
        return retval
    else:
        return string

def remove_boxed(string: str) -> Optional[str]:
    left = "\\boxed{"
    if string[: len(left)] == left:
        return string[len(left) : -1]
    else:
        return string
    
def remove_text(string: str) -> Optional[str]:
    left = "\\text{"
    if string[: len(left)] == left:
        return string[len(left) : -1]
    else:
        return string
    
class Prompt:
    def __init__(self, env: str = "click-button") -> None:
        self.llm = "chatgpt"
        self.davinci_type_regex = "^type\s.{1,}$"
        self.chatgpt_type_regex = '^type\s[^"]{1,}$'
        self.press_regex = (
            "^press\s(enter|arrowleft|arrowright|arrowup|arrowdown|backspace)$"
        )
        self.clickxpath_regex = "^clickxpath\s.{1,}$"
        self.clickoption_regex = "^clickoption\s.{1,}$"
        self.movemouse_regex = "^movemouse\s.{1,}$"

        if os.path.exists(f"prompt/{env}/"):
            base_dir = f"prompt/{env}/"
        else:
            base_dir = f"prompt/"

        with open(base_dir + "example.txt") as f:
            self.example_prompt = f.read()

        with open(base_dir + "first_action.txt") as f:
            self.first_action_prompt = f.read()

        with open(base_dir + "base.txt") as f:
            self.base_prompt = f.read()
            self.base_prompt = self.replace_regex(self.base_prompt)

        with open(base_dir + "initialize_plan.txt") as f:
            self.init_plan_prompt = f.read()

        with open(base_dir + "action.txt") as f:
            self.action_prompt = f.read()

        with open(base_dir + "rci_action.txt") as f:
            self.rci_action_prompt = f.read()
            self.rci_action_prompt = self.replace_regex(self.rci_action_prompt)

        with open(base_dir + "update_action.txt") as f:
            self.update_action = f.read()

    def replace_regex(self, base_prompt):
        if self.llm == "chatgpt":
            base_prompt = base_prompt.replace("{type}", self.chatgpt_type_regex)
        elif self.llm == "davinci":
            base_prompt = base_prompt.replace("{type}", self.davinci_type_regex)
        else:
            raise NotImplemented

        base_prompt = base_prompt.replace("{press}", self.press_regex)
        base_prompt = base_prompt.replace("{clickxpath}", self.clickxpath_regex)
        base_prompt = base_prompt.replace("{clickoption}", self.clickoption_regex)
        base_prompt = base_prompt.replace("{movemouse}", self.movemouse_regex)

        return base_prompt

def _get_html_state(problem, states):
    extra_html_task = [
        "click-dialog",
        "click-dialog-2",
        "use-autocomplete",
        "choose-date",
    ]

    html_body = states[0].html_body
    if problem in extra_html_task:
        html_body += states[0].html_extra
    return html_body


def _convert_to_miniwob_action(instruction: str):
    instruction = instruction.split(" ")
    inst_type = instruction[0]
    inst_type = inst_type.lower()

    if inst_type == "type":
        characters = " ".join(instruction[1:])
        characters = characters.replace('"', "")
        return MiniWoBType(characters)
    elif inst_type == "clickid":
        element_id = " ".join(instruction[1:])
        return MiniWoBElementClickId(element_id)
    elif inst_type == "press":
        key_type = instruction[1].lower()
        if key_type == "enter":
            return MiniWoBType("\n")
        elif key_type == "space":
            return MiniWoBType(" ")
        elif key_type == "arrowleft":
            return MiniWoBType(Keys.LEFT)
        elif key_type == "arrowright":
            return MiniWoBType(Keys.RIGHT)
        elif key_type == "backspace":
            return MiniWoBType(Keys.BACKSPACE)
        elif key_type == "arrowup":
            return MiniWoBType(Keys.UP)
        elif key_type == "arrowdown":
            return MiniWoBType(Keys.DOWN)
        else:
            raise NotImplemented
    elif inst_type == "movemouse":
        xpath = " ".join(instruction[1:])
        return MiniWoBMoveXpath(xpath)
    elif inst_type == "clickxpath":
        xpath = " ".join(instruction[1:])
        return MiniWoBElementClickXpath(xpath)
    elif inst_type == "clickoption":
        xpath = " ".join(instruction[1:])
        return MiniWoBElementClickOption(xpath)
    else:
        raise ValueError("Invalid instruction")

class MiniWobUserProxyAgent(ResponsiveAgent):
    """(Experimental) A agent that can handle online decision making in miniwob+ benchmark."""

    MAX_CONSECUTIVE_AUTO_REPLY = (
        15  # maximum number of consecutive auto replies (subject to future change)
    )

    def __init__(
        self,
        name= "MinWobAgent",
        is_termination_msg = lambda x: "terminate" in x.get("content").lower(),  
        max_consecutive_auto_reply: Optional[int] = None,
        human_input_mode: Optional[str] = "NEVER",
        function_map: Optional[Dict[str, Callable]] = None,
        code_execution_config: Optional[Union[Dict, bool]] = None,
        oai_config: Optional[Union[Dict, bool]] = False,
        system_message: Optional[str] = "",
        problem=None,
        headless=False,
        **kwargs,
    ):
    
        super().__init__(
            name=name,
            is_termination_msg=is_termination_msg,
            max_consecutive_auto_reply = max_consecutive_auto_reply,
            human_input_mode = human_input_mode,
            function_map = function_map,
            code_execution_config = code_execution_config,
            oai_config = oai_config,
            system_message = system_message,
            **kwargs,
        )

        self.problem = problem
        self.headless = headless
        self.current_plan = ""
        self.past_instruction = []
        with open("config.json") as config_file:
            api_key = json.load(config_file)["api_key"]
            openai.api_key = api_key
        self.llm = "chatgpt"
        self.model = "gpt-3.5-turbo"

        self.prompt = Prompt(env=problem)
        self.env = gym.make(
            "MiniWoBEnv-v0", env_name=self.problem, headless=self.headless
        )
        states = self.env.reset(seeds=[random.random()], record_screenshots=True)
        self.task = states[0].utterance
        html_state = _get_html_state(self.problem, states)
        self.html_state = html_state  

        # others
        self.identify_plan = False
        self.get_plan = True

        self.identify_action = False
        self.ask_action = True

        self.rci_plan_loop = 0
        self.unexecuted_steps = 0
        self.pt = None
        self.rci_action = 1

        # succeed
        self.success = 0
        
    def _webpage_state_prompt(self, initial: bool = False): 
        pt = "\n\n"
        pt += "Below is the HTML code of the webpage where the agent should solve a task.\n"
        pt += self.html_state
        pt += "\n\n"
        if self.prompt.example_prompt and initial:
            pt += self.prompt.example_prompt
            pt += "\n\n"

        pt += "Current task: "
        pt += self.task
        pt += "\n"

        return pt

    def generate_init_message(self):

        super().reset()
        pt = self.prompt.base_prompt
        pt = self._webpage_state_prompt(initial=True)
        pt += self.prompt.init_plan_prompt
        return pt

    def _current_plan_prompt(self):
        pt = "\n\n"
        pt += "Here is a plan you are following now.\n"
        pt += f"{self.current_plan}"
        pt += "\n\n"

        return pt

    def _instruction_history_prompt(self):
        pt = "\n\n"
        pt += "We have a history of instructions that have been already executed by the autonomous agent so far.\n"
        if not self.past_instruction:
            pt += "No instruction has been executed yet."
        else:
            for idx, inst in enumerate(self.past_instruction):
                pt += f"{idx+1}: "
                pt += inst
                pt += "\n"
        pt += "\n\n"

        return pt

    def rci_action(self, instruciton: str, pt=None):
        instruciton = self._process_instruction(instruciton)

        loop_num = 0
        while self._check_regex(instruciton):
            if loop_num >= self.rci_limit:
                print(instruciton)
                raise ValueError("Action RCI failed")

            pt += self.prompt.rci_action_prompt
            instruciton = self.get_response(pt)

            pt += instruciton
            instruciton = self._process_instruction(instruciton)

            loop_num += 1

        return pt, instruciton

    def ask_action_prompt(self):
        pt = self.prompt.base_prompt
        pt += self._webpage_state_prompt()
        if self.prompt.init_plan_prompt:
            pt += self._current_plan_prompt()
        pt += self._instruction_history_prompt()
        if self.past_instruction:
            update_action_prompt = self.prompt.action_prompt.replace(
                "{prev_inst}", self.past_instruction[-1]
            )
            if len(self.past_instruction) == 1:
                update_action_prompt = self.prompt.action_prompt.replace(
                    "{order}", "2nd"
                )
            elif len(self.past_instruction) == 2:
                update_action_prompt = self.prompt.action_prompt.replace(
                    "{order}", "3rd"
                )
            else:
                update_action_prompt = self.prompt.action_prompt.replace(
                    "{order}", f"{len(self.past_instruction)+1}th"
                )

            action_prompt = update_action_prompt
        else:
            action_prompt = self.prompt.first_action_prompt
        action_prompt = ("Please put the instruction in \\boxed{} in your reply and do not adding other characters. \
        based on the plan,"+ action_prompt)
        pt += action_prompt

        return pt

    def _check_regex(self, instruciton):
        return (
            (not re.search(self.prompt.clickxpath_regex, instruciton, flags=re.I))
            and (not re.search(self.prompt.chatgpt_type_regex, instruciton, flags=re.I))
            and (not re.search(self.prompt.davinci_type_regex, instruciton, flags=re.I))
            and (not re.search(self.prompt.press_regex, instruciton, flags=re.I))
            and (not re.search(self.prompt.clickoption_regex, instruciton, flags=re.I))
            and (not re.search(self.prompt.movemouse_regex, instruciton, flags=re.I))
        )

    def _process_instruction(self, instruciton: str):
        end_idx = instruciton.find("`")
        if end_idx != -1:
            instruciton = instruciton[:end_idx]

        instruciton = instruciton.replace("`", "")
        instruciton = instruciton.replace("\n", "")
        instruciton = instruciton.replace("\\n", "\n")
        instruciton = instruciton.strip()
        instruciton = instruciton.strip("'")

        return instruciton
    
    def save_result(self, value):
        path_dir = os.path.join("./result", self.problem+".json")
        if os.path.exists(path_dir):
            with open(path_dir, 'r') as f:
                data = json.load(f)
        else:
            data = {}

        if 'value' in data:
            if value >0:
                data['value'] += 1
        else:
            data['value'] = 0

        with open(path_dir, 'w') as f:
            json.dump(data, f)
        
    def generate_reply(self, messages: List[Dict], default_reply: Union[str, Dict] = "") -> Union[str, Dict]:
        messages = messages[-1]
        messages = messages.get("content", "")
        if not self.identify_plan:
            if self.get_plan:
                self.get_plan = False
                self.current_plan = messages
                reply = "According to the provided example plans, find problems with this plan for the given task. \
                Based on your findings, just tell me what is the plan for the agent to complete the task? \
                If the previous plan is right, repeat your previous answer again."
                return reply
            else:
                self.identify_plan = True
                self.current_plan = messages
                step = 1
                while True:
                    if (str(step) + ".") not in messages:
                        break
                    else:
                        step+=1
                step -=1
                self.unexecuted_steps += step
        if self.unexecuted_steps != 0: 
            reply = ""
            if not self.identify_action:
                self.action_pt = self.ask_action_prompt()
                reply += self.action_pt
                self.identify_action = True
                return reply
            else:
                self.unexecuted_steps =  self.unexecuted_steps - 1
                self.identify_action = False
                messages = last_boxed_only_string(messages)
                messages = remove_boxed(messages)
                messages = remove_text(messages)
                self.action_pt += self._process_instruction(messages) + "`."
                self.instruciton = self._process_instruction(messages)
                self.past_instruction.append(self.instruciton)
                try:
                    miniwob_action = _convert_to_miniwob_action(self.instruciton)
                    states, rewards, dones, _ = self.env.step([miniwob_action])
                except ValueError:
                    reply += (
                        "Unsuccess! Please return TERMINATE"
                    )
                    rewards = [0]
                    self.save_result(-1)
                    return reply
                if self.unexecuted_steps != 0:
                    if rewards[0] > 0: 
                        reply = "Success! Everything is done now. Please return TERMINATE"
                        self.save_result(1)
                        return reply
                    elif rewards[0] < 0: # bug 
                        reply = "Unsuccess! Please return TERMINATE"
                        self.save_result(-1)
                        return reply
                    else:
                        html_state = _get_html_state(self.problem, states)
                        self.html_state = html_state 
                else:
                    if rewards[0] > 0:
                        reply = "Success! Everything is done now. Please return TERMINATE"
                        self.save_result(1)
                    else:
                        reply = "Unsuccess! Please return TERMINATE"
                        self.save_result(-1)
                return reply
                
