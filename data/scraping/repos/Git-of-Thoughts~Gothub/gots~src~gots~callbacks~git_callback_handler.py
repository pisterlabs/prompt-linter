import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Union

from git import Repo
from langchain.callbacks.base import BaseCallbackHandler
from langchain.schema.agent import AgentAction, AgentFinish
from langchain.schema.output import LLMResult


# First, define custom callback handler implementations
class GitCallbackHandler(BaseCallbackHandler):
    ##This class is a list of callback functions to be used in Langchain callbacks and to be used in the Gothub directory
    """
    FIXME Tool callbacks seem to have bugs.
    """

    def __init__(self, repo: Repo):
        self.repo = repo  # This will store the repo object
        logfile = (
            Path(repo.working_dir).parent / "log.txt"
        )  # This will store the path to the log file
        self.logfilepath = logfile  # This will store the path to the log file
        with open(self.logfilepath, "a") as f:  # Open the file in append mode
            f.write("## Logging: init\n")  # Write the message to the file

    def write_to_log(self, message: str):
        with open(self.logfilepath, "a") as f:  # Open the file in append mode
            f.write(message)  # Write the message to the file

    def on_tool_start(
        self, serialized: Dict[str, Any], input_str: str, **kwargs: Any
    ) -> Any:
        ##This function is called when the tool starts running
        """Run when tool starts running."""
        serialized_str = json.dumps(serialized, indent=4)  # Indent to format JSON
        # kwargs_str = json.dumps(kwargs.toJSON(), indent=4)  # Indent to format JSON
        self.write_to_log(
            "## Logging: on_tool_start\n"
            + "### Serialized:\n"
            + serialized_str
            + "\n"
            + "### Input String:\n"
            + input_str
            + "\n"
        )  # Write the message to the file
        self.repo.git.add(A=True)  # This will add all files to the staging area
        self.repo.git.commit("-m", "on tool start", "--allow-empty")

    def on_llm_start(
        self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any
    ) -> Any:
        ##This function is called when the llm starts running
        serialized_str = json.dumps(serialized, indent=4)  # Indent to format JSON
        # kwargs_str = json.dumps(**kwargs.toJSON(), indent=4)  # Indent to format JSON
        prompts_str = "\n".join(prompts)
        self.write_to_log(
            "## Logging: on_llm_start\n"
            + "### Serialized:\n"
            + serialized_str
            + "\n"
            + "### Prompts:\n"
            + prompts_str
            + "\n"
        )  # Write the message to the file
        self.repo.git.add(A=True)  # This will add all files to the staging area
        self.repo.git.commit("-m", "on llm start", "--allow-empty")

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> Any:
        args_str = str(response)  # Indent to format JSON
        # kwargs_str = json.dumps(**kwargs.toJSON(), indent=4)  # Indent to format JSON
        self.write_to_log("## Logging: on_llm_end\n" + args_str + "\n")
        self.repo.git.add(A=True)  # This will add all files to the staging area
        self.repo.git.commit("-m", "on llm end", "--allow-empty")

    def on_llm_error(
        self, error: Union[Exception, KeyboardInterrupt], **kwargs: Any
    ) -> Any:
        """Run when on_llm_error."""
        error_str = str(error)  # Convert the exception to a string
        # kwargs_str = json.dumps(kwargs, indent=4)  # Indent to format JSON

        self.write_to_log(
            "## Logging: on_llm_error\n"
            + "### Error:\n"
            + error_str
            + "\n"
            # + "Kwargs:\n"
            # + kwargs_str
            # + "\n"
        )
        self.repo.git.add(A=True)  # This will add all files to the staging area
        self.repo.git.commit("-m", "on llm error", "--allow-empty")

    def on_chat_model_start(
        self, serialized: Dict[str, Any], messages: Any, **kwargs: Any
    ) -> Any:
        """Run when chat starts running."""
        serialized_str = json.dumps(serialized, indent=4)  # Indent to format JSON
        messages_str = str(messages)  # Indent to format JSON
        self.write_to_log(
            "## Logging: on_chat_model_start\n"
            + "### Serialized:\n"
            + serialized_str
            + "\n"
            + "### message:\n"
            + messages_str
            + "\n"
        )
        self.repo.git.add(A=True)  # This will add all files to the staging area
        self.repo.git.commit("-m", "on chat model start", "--allow-empty")

    def on_llm_new_token(self, *args, **kwargs: Any) -> Any:
        """Run when tool starts running."""
        args_str = json.dumps(*args.toJSON(), indent=4)  # Indent to format JSON
        kwargs_str = json.dumps(**kwargs.toJSON(), indent=4)  # Indent to format JSON
        self.write_to_log(
            "## Logging: on_llm_new_token\n" + args_str + "\n" + kwargs_str + "\n"
        )
        self.repo.git.add(A=True)  # This will add all files to the staging area
        self.repo.git.commit("-m", "on llm new token", "--allow-empty")

    def on_tool_end(self, output: str, **kwargs: Any) -> Any:
        """Run when tool ends running."""
        # args_str = json.dumps(*args.toJSON(), indent=4)  # Indent to format JSON
        # kwargs_str = json.dumps(**kwargs.toJSON(), indent=4)  # Indent to format JSON
        self.write_to_log("## Logging: on_tool_end\n" + output)
        self.repo.git.add(A=True)  # This will add all files to the staging area
        self.repo.git.commit("-m", "on tool end", "--allow-empty")

    def on_tool_error(
        self, error: Union[Exception, KeyboardInterrupt], **kwargs: Any
    ) -> Any:
        """Run when tool errors."""
        error_str = str(error)  # Convert the exception to a string
        # kwargs_str = json.dumps(kwargs, indent=4)  # Indent to format JSON

        self.write_to_log(
            "## Logging: on_tool_error\n"
            + "### Error:\n"
            + error_str
            + "\n"
            # + "Kwargs:\n"
            # + kwargs_str
            # + "\n"
        )

        self.repo.git.add(A=True)  # This will add all files to the staging area
        self.repo.git.commit("-m", "on tool error", "--allow-empty")

    def on_text(self, text: str, **kwargs: Any) -> Any:
        """Run on arbitrary text."""
        # args_str = json.dumps(*args.toJSON(), indent=4)  # Indent to format JSON
        # kwargs_str = json.dumps(**kwargs.toJSON(), indent=4)  # Indent to format JSON
        self.write_to_log("## Logging: on_text\n" + "### Text:\n" + text + "\n")
        self.repo.git.add(A=True)  # This will add all files to the staging area
        self.repo.git.commit("-m", "on text", "--allow-empty")

    def on_chain_start(
        self, serialized: Dict[str, Any], inputs: Dict[str, Any], **kwargs: Any
    ) -> Any:
        """Run when chain starts running."""
        serialized_str = json.dumps(serialized, indent=4)  # Indent to format JSON
        # kwargs_str = json.dumps(**kwargs.toJSON(), indent=4)  # Indent to format JSON
        inputs_str = json.dumps(inputs, indent=4)
        self.write_to_log(
            "## Logging: on_chain_start\n"
            + "### Serialized:\n"
            + serialized_str
            + "\n"
            + "### inputs:\n"
            + inputs_str
            + "\n"
        )
        self.repo.git.add(A=True)  # This will add all files to the staging area
        self.repo.git.commit("-m", "on chain start", "--allow-empty")

    def on_chain_end(self, outputs: Dict[str, Any], **kwargs: Any) -> Any:
        """Run when chain ends running."""
        args_str = json.dumps(outputs, indent=4)  # Indent to format JSON
        # kwargs_str = json.dumps(**kwargs.toJSON(), indent=4)  # Indent to format JSON
        self.write_to_log("## Logging: on_chain_end\n" + args_str + "\n")
        self.repo.git.add(A=True)  # This will add all files to the staging area
        self.repo.git.commit("-m", "on chain end", "--allow-empty")

    def on_chain_error(
        self, error: Union[Exception, KeyboardInterrupt], **kwargs: Any
    ) -> Any:
        """Run when chain errors running."""
        error_str = str(error)  # Convert the exception to a string
        # kwargs_str = json.dumps(kwargs, indent=4)  # Indent to format JSON

        self.write_to_log(
            "## Logging: on_chain_error\n"
            + "### Error:\n"
            + error_str
            + "\n"
            # + "Kwargs:\n"
            # + kwargs_str
            # + "\n"
        )
        self.repo.git.add(A=True)  # This will add all files to the staging area
        self.repo.git.commit("-m", "on chain error", "--allow-empty")

    def on_agent_finish(self, finish: AgentFinish, **kwargs: Any) -> Any:
        """Run on agent end."""
        args_str = json.dumps(finish.return_values, indent=4)  # Indent to format JSON
        # kwargs_str = json.dumps(**kwargs.toJSON(), indent=4)  # Indent to format JSON
        args_add = finish.log
        self.write_to_log(
            "## Logging: on_agent_finish\n"
            + "### Return values:\n"
            + args_str
            + "\n"
            + "### Additional logs:\n"
            + args_add
        )
        self.repo.git.add(A=True)  # This will add all files to the staging area
        self.repo.git.commit("-m", "on agent finish", "--allow-empty")

    def on_agent_action(self, action: AgentAction, **kwargs: Any) -> Any:
        """Run on agent action."""
        action_tool = action.tool
        action_tool_input = json.dumps(action.tool_input, indent=4)
        action_log = action.log
        # kwargs_str = json.dumps(**kwargs.toJSON(), indent=4)  # Indent to format JSON
        self.write_to_log(
            "## Logging: on_agent_action\n"
            + "### Tool used:\n"
            + action_tool
            + "\n"
            + "### Tool input:\n"
            + action_tool_input
            + "\n"
            + "### Additional log:\n"
            + action_log
            + "\n"
        )
        self.repo.git.add(A=True)  # This will add all files to the staging area
        self.repo.git.commit("-m", "on agent action", "--allow-empty")
