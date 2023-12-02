from .adot_conversable_agent import AdotConversableAgent
from typing import Callable, Dict, Optional, Union
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union
from .agent import Agent
import json
try:
    from termcolor import colored
except ImportError:

    def colored(x, *args, **kwargs):
        return x

class AdotUserProxyAgent(AdotConversableAgent):
    """(In preview) A proxy agent for the user, that can execute code and provide feedback to the other agents.

    UserProxyAgent is a subclass of ConversableAgent configured with `human_input_mode` to ALWAYS
    and `llm_config` to False. By default, the agent will prompt for human input every time a message is received.
    Code execution is enabled by default. LLM-based auto reply is disabled by default.
    To modify auto reply, register a method with [`register_reply`](conversable_agent#register_reply).
    To modify the way to get human input, override `get_human_input` method.
    To modify the way to execute code blocks, single code block, or function call, override `execute_code_blocks`,
    `run_code`, and `execute_function` methods respectively.
    To customize the initial message when a conversation starts, override `generate_init_message` method.
    """

    def __init__(
        self,
        name: str,
        is_termination_msg: Optional[Callable[[Dict], bool]] = None,
        max_consecutive_auto_reply: Optional[int] = None,
        human_input_mode: Optional[str] = "ALWAYS",
        function_map: Optional[Dict[str, Callable]] = None,
        code_execution_config: Optional[Union[Dict, bool]] = None,
        default_auto_reply: Optional[Union[str, Dict, None]] = "",
        llm_config: Optional[Union[Dict, bool]] = False,
        system_message: Optional[str] = "",
        database=None
    ):
        """
        Args:
            name (str): name of the agent.
            is_termination_msg (function): a function that takes a message in the form of a dictionary
                and returns a boolean value indicating if this received message is a termination message.
                The dict can contain the following keys: "content", "role", "name", "function_call".
            max_consecutive_auto_reply (int): the maximum number of consecutive auto replies.
                default to None (no limit provided, class attribute MAX_CONSECUTIVE_AUTO_REPLY will be used as the limit in this case).
                The limit only plays a role when human_input_mode is not "ALWAYS".
            human_input_mode (str): whether to ask for human inputs every time a message is received.
                Possible values are "ALWAYS", "TERMINATE", "NEVER".
                (1) When "ALWAYS", the agent prompts for human input every time a message is received.
                    Under this mode, the conversation stops when the human input is "exit",
                    or when is_termination_msg is True and there is no human input.
                (2) When "TERMINATE", the agent only prompts for human input only when a termination message is received or
                    the number of auto reply reaches the max_consecutive_auto_reply.
                (3) When "NEVER", the agent will never prompt for human input. Under this mode, the conversation stops
                    when the number of auto reply reaches the max_consecutive_auto_reply or when is_termination_msg is True.
            function_map (dict[str, callable]): Mapping function names (passed to openai) to callable functions.
            code_execution_config (dict or False): config for the code execution.
                To disable code execution, set to False. Otherwise, set to a dictionary with the following keys:
                - work_dir (Optional, str): The working directory for the code execution.
                    If None, a default working directory will be used.
                    The default working directory is the "extensions" directory under
                    "path_to_autogen".
                - use_docker (Optional, list, str or bool): The docker image to use for code execution.
                    If a list or a str of image name(s) is provided, the code will be executed in a docker container
                    with the first image successfully pulled.
                    If None, False or empty, the code will be executed in the current environment.
                    Default is True, which will be converted into a list.
                    If the code is executed in the current environment,
                    the code must be trusted.
                - timeout (Optional, int): The maximum execution time in seconds.
                - last_n_messages (Experimental, Optional, int): The number of messages to look back for code execution. Default to 1.
            default_auto_reply (str or dict or None): the default auto reply message when no code execution or llm based reply is generated.
            llm_config (dict or False): llm inference configuration.
                Please refer to [Completion.create](/docs/reference/oai/completion#create)
                for available options.
                Default to false, which disables llm-based auto reply.
            system_message (str): system message for ChatCompletion inference.
                Only used when llm_config is not False. Use it to reprogram the agent.
        """
        super().__init__(
            name,
            system_message,
            is_termination_msg,
            max_consecutive_auto_reply,
            human_input_mode,
            function_map,
            code_execution_config,
            llm_config,
            default_auto_reply,
            database=database,
        )

        ## override
        self._reply_func_list = []
        self.register_reply([Agent, None], AdotUserProxyAgent.generate_oai_reply)
        self.register_reply([Agent, None], AdotUserProxyAgent.generate_code_execution_reply)
        self.register_reply([Agent, None], AdotUserProxyAgent.generate_function_call_reply)
        self.register_reply([Agent, None], AdotUserProxyAgent.check_termination_and_human_reply)

    def generate_function_call_reply(
        self,
        session_id:str,
        messages: Optional[List[Dict]] = None,
        sender: Optional[Agent] = None,
        config: Optional[Any] = None,
        **kwargs,
    ):
        """Generate a reply using function call."""
        if config is None:
            config = self
        if messages is None:
            messages = self._oai_messages.query(session_id, agent=sender)
        message = messages[-1]
        if "function_call" in message:
            _, func_return = self.execute_function(message["function_call"], session_id=session_id)
            return True, func_return
        return False, None
    
    def execute_function(self, func_call, session_id=None):
        """Execute a function call and return the result.

        Override this function to modify the way to execute a function call.

        Args:
            func_call: a dictionary extracted from openai message at key "function_call" with keys "name" and "arguments".

        Returns:
            A tuple of (is_exec_success, result_dict).
            is_exec_success (boolean): whether the execution is successful.
            result_dict: a dictionary with keys "name", "role", and "content". Value of "role" is "function".
        """
        func_name = func_call.get("name", "")
        func = self._function_map.get(func_name, None)

        is_exec_success = False
        if func is not None:
            # Extract arguments from a json-like string and put it into a dict.
            input_string = self._format_json_str(func_call.get("arguments", "{}"))
            try:
                arguments = json.loads(input_string)
            except json.JSONDecodeError as e:
                arguments = None
                content = f"Error: {e}\n You argument should follow json format."

            # Try to execute the function
            if arguments is not None:
                print(
                    colored(f"\n>>>>>>>> EXECUTING FUNCTION {func_name}...", "magenta"),
                    flush=True,
                )
                try:
                    content = func(session_id=session_id, **arguments)
                    is_exec_success = True
                except Exception as e:
                    content = f"Error: {e}"
        else:
            content = f"Error: Function {func_name} not found."

        return is_exec_success, {
            "name": func_name,
            "role": "function",
            "content": str(content),
        }
