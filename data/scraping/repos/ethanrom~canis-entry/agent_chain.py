from langchain.chains import LLMChain
from langchain.agents import ZeroShotAgent, AgentExecutor
from langchain.llms import OpenAI

from memory import memory
from tools import zeroshot_tools

import config
import os




os.environ["OPENAI_API_KEY"] = config.OPENAI_API_KEY
temperature = 0


CUSTOM_FORMAT_INSTRUCTIONS = """Use the following format:

User Input: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question"""


ZEROSHOT_PREFIX = f"""
First 3 rows of the dataset:
Name (English),Name (Chinese),Region of Focus,Language,Entity owner (English),Entity owner (Chinese),Parent entity (English),Parent entity (Chinese),X (Twitter) handle,X (Twitter) URL,X (Twitter) Follower #,Facebook page,Facebook URL,Facebook Follower #,Instragram page,Instagram URL,Instagram Follower #,Threads account,Threads URL,Threads Follower #,YouTube account,YouTube URL,YouTube Subscriber #,TikTok account,TikTok URL,TikTok Subscriber #
Yang Xinmeng (Abby Yang),杨欣萌,Anglosphere,English,China Media Group (CMG),中央广播电视总台,Central Publicity Department,中共中央宣传部,_bubblyabby_,https://twitter.com/_bubblyabby_,1678.00,itsAbby-103043374799622,https://www.facebook.com/itsAbby-103043374799622,1387432.00,_bubblyabby_,https://www.instagram.com/_bubblyabby_/,9507.00,_bubblyabby_,https://www.threads.net/@_bubblyabby_,197.00,itsAbby,https://www.youtube.com/itsAbby,4680.00,_bubblyabby_,https://www.tiktok.com/@_bubblyabby_,660.00
CGTN Culture Express,,Anglosphere,English,China Media Group (CMG),中央广播电视总台,Central Publicity Department,中共中央宣传部,_cultureexpress,https://twitter.com/_cultureexpress,2488.00,,,,_cultureexpress/,https://www.instagram.com/_cultureexpress/,635.00,,,,,,,,,


====
You have access to the following tools:"""


ZEROSHOT_SUFFIX = """Begin"

{chat_history}
Customer: {input}
{agent_scratchpad}"""



def get_agent_chain():

    prompt = ZeroShotAgent.create_prompt(
        zeroshot_tools,
        prefix=ZEROSHOT_PREFIX,
        suffix=ZEROSHOT_SUFFIX,
        format_instructions=CUSTOM_FORMAT_INSTRUCTIONS,
        input_variables=["input", "chat_history", "agent_scratchpad"]
    )

    zeroshot_agent_llm = OpenAI(temperature=temperature, streaming=True)
    zeroshot_llm_chain = LLMChain(llm=zeroshot_agent_llm, prompt=prompt)
    zeroshot_agent = ZeroShotAgent(llm_chain=zeroshot_llm_chain)
    zeroshot_agent_chain = AgentExecutor.from_agent_and_tools(agent=zeroshot_agent, tools=zeroshot_tools, verbose=True, memory=memory, handle_parsing_errors=True)
    return zeroshot_agent_chain