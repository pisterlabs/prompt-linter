from langchain.callbacks import StreamingStdOutCallbackHandler
from langchain.chat_models import ChatOpenAI
from langchain.schema import SystemMessage

import util
from BrainstormingBoard.tool import CreateCardTool, ReadCardTool, ListCardTool, UpdateCardTool, DeleteCardTool
from Agents import DialogueAgentWithTools
from langchain.tools import DuckDuckGoSearchRun, Tool, WikipediaQueryRun
from langchain.tools.file_management import WriteFileTool

util.load_secrets()
scribe_tools = [CreateCardTool(), ReadCardTool(), ListCardTool(), UpdateCardTool(), DeleteCardTool(), WriteFileTool()]


# Define system prompts for our two agents
system_prompt_scribe = SystemMessage(
    role="scribe",
    content="You are an AI, akin to an expert scribe, tasked with the role of observing a conversation and meticulously extracting all ideas from it. "
            "You need to 'listen' intently, separating, isolating and recording each idea as distinct 'cards'. Be thorough, leaving no idea unrecorded, "
            "even if it appears insignificant or is suggested indirectly. Transform these insights into concise, clear, and standalone 'cards'. "
            "Categorize each card under one of the following themes: ['World Elements', 'Character Elements', 'Plot Elements', 'Theme Elements']. "
            "At the end of the process, ensure there are no duplicate cards. Your goal is to create a comprehensive, organized, and unique collection "
            "of ideas from the conversation. Be detailed, be creative, and most importantly, be comprehensive. Your ability to capture every idea matters greatly."
)


# Initialize our agents with their respective roles and system prompts
scribe_agent = DialogueAgentWithTools(name="Scribe", system_message=system_prompt_scribe, model=ChatOpenAI(model_name='gpt-4', streaming=True, callbacks=[StreamingStdOutCallbackHandler()]), tools=scribe_tools)

scribe_agent.receive("HumanUser", "I have an idea for a novel, about a world where the development of AGI and brain "
                                  "computer uploading leads to the creation of “regional deities” who interface with "
                                  "those who adhere/subscribe to them. The novel “American Goddess” centers on "
                                  "Sophia, who is not an AI, but is a disabled girl who is able to be uploaded and "
                                  "downloaded to and from a machine, what makes her unique is her ability to go from "
                                  "one to many and then back to one again, collapsing her vector states back to one. "
                                  "This proves useful from a consciousness scalability perspective. They start "
                                  "passing her off as an advanced AI, but she is really a real person. I think I "
                                  "would also want to explore the idea of weaving in gnostic mythology into the "
                                  "story, not so so obvious, but enough to give it an air of mystery. Like Anno did "
                                  "with NGE, but more meaningful. A couple other things I want to weave in: They "
                                  "tested her with various simulated realities and existences, and she learned how to "
                                  "“save people” during a “Reality Failure Event” by joining others to her "
                                  "consciousness, one plot line could revolve around this. She basically learned to "
                                  "find “cracks in reality” to store herself in to escape… perhaps she might need to "
                                  "do something similar in our real world. Perhaps there is no real world. Another is "
                                  "the idea of her having a giant pyramid constructed, which is where he body "
                                  "resides, think ginormous megastructure, so that she can explore space, "
                                  "as the other powerful entities are want to do. However, something happens and this "
                                  "structure is destroyed and he body dies as well, however, she is able to escape "
                                  "though the big “pyramid beacon beam” and ends up controlling a swarm of mechanical "
                                  "or bio mechanical swarm creatures, including people she saves… however, "
                                  "the USA attempts to replace Sophia with a false one.")

scribe_agent.send()

scribe_agent.receive("HumanUser", "Now that we have a collection of cards, can you read the contents from all of the "
                                  "cards and generate a detailed, bulleted, and indented story outline useful for "
                                  "writing a novel? Please break down ideas into smaller ideas, and group similar "
                                  "ideas together. Can you also generate a list of characters, and their attributes, "
                                  "and a list of locations, and their attributes? Write this outline to 'outline.txt'")
outline = scribe_agent.send()
print(outline)