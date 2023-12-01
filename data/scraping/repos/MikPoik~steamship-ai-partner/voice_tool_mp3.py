"""Tool for generating images."""
from steamship import Steamship  #upm package(steamship)
from steamship.agents.llms import OpenAI  #upm package(steamship)
from steamship.agents.tools.speech_generation import GenerateSpeechTool  #upm package(steamship)
from steamship.agents.utils import with_llm  #upm package(steamship)
from steamship.utils.repl import ToolREPL  #upm package(steamship)
from typing import Any, List, Union
from steamship import Block, Task  #upm package(steamship)
from steamship.agents.schema import AgentContext  #upm package(steamship)
from tools.active_companion import VOICE_ID  #upm package(steamship)


class VoiceToolMP3(GenerateSpeechTool):
    """Tool to generate audio from text."""

    name: str = "GenerateSpokenAudio"
    human_description: str = "Generates spoken audio from text."
    agent_description: str = (
        "Use this tool to generate spoken audio from text, the input should be a plain text string containing the "
        "content to be spoken.")

    prompt_template = ("{subject}")

    def run(self, tool_input: List[Block],
            context: AgentContext) -> Union[List[Block], Task[Any]]:

        modified_inputs = [
            Block(text=self.prompt_template.format(subject=block.text))
            for block in tool_input
        ]
        speech = GenerateSpeechTool()
        speech.generator_plugin_config = {"voice_id": VOICE_ID}
        return speech.run(modified_inputs, context)


if __name__ == "__main__":
    tool = VoiceToolMP3()
    with Steamship.temporary_workspace() as client:
        ToolREPL(tool).run_with_client(
            client=client, context=with_llm(llm=OpenAI(client=client)))
