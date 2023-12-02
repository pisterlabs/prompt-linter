"""Tool for generating audio"""
from steamship import Steamship  #upm package(steamship)
from steamship.agents.llms import OpenAI  #upm package(steamship)
from steamship.agents.tools.speech_generation import GenerateSpeechTool  #upm package(steamship)
from steamship.agents.utils import with_llm  #upm package(steamship)
from steamship.utils.repl import ToolREPL  #upm package(steamship)
from typing import Any, List, Union
from steamship import Block, Task  #upm package(steamship)
from steamship.agents.schema import AgentContext
from steamship.agents.schema import AgentContext, Tool  #upm package(steamship)
#from tools.active_companion import VOICE_ID #upm package(steamship)


class UnrealSpeechTool(Tool):
  """Tool to generate audio from text."""

  name: str = "GenerateSpokenAudio"
  human_description: str = "Generates spoken audio from text."
  agent_description: str = (
      "Use this tool to generate spoken audio from text, the input should be a plain text string containing the "
      "content to be spoken.")

  prompt_template = ("{subject}")
  generator_plugin_handle: str = "unrealspeechdev"
  generator_plugin_config: dict = {
      "unrealspeech_api_key":
      "zrNyr6DUwiHL2dh2WJE4cykW6xKjhyD4mDN5vzrjYz7JdDNF3W316H",
      "temperature": 0.5,
      "pitch": 1.02,
      "speed": 0.1
  }

  def run(self, tool_input: List[Block],
          context: AgentContext) -> Union[List[Block], Task[Any]]:

    generator = context.client.use_plugin(self.generator_plugin_handle,
                                          config=self.generator_plugin_config)
    text_input = ""
    for block in tool_input:
      if not block.is_text():
        continue
      text_input += block.text

    task = generator.generate(text=text_input,
                              make_output_public=True,
                              append_output_to_file=True)
    task.wait()
    blocks = task.output.blocks
    output_blocks = []
    for block in blocks:
      output_blocks.append(block)
    return output_blocks
    return output_blocks


if __name__ == "__main__":
  tool = UnrealSpeechTool()
  with Steamship.temporary_workspace() as client:
    ToolREPL(tool).run_with_client(client=client,
                                   context=with_llm(llm=OpenAI(client=client)))
