from typing import Optional, List, Dict

from langchain.memory import ConversationBufferMemory
from langchain.agents import AgentType, initialize_agent, load_tools
from langchain.input import print_text
from langchain.callbacks.base import CallbackManager

from llm_oracle import llm
from llm_oracle.tools.search import get_search_tool
from llm_oracle.tools.read_link import get_read_link_tool
from llm_oracle.agents.base import OracleAgent
from llm_oracle.agents.output import OUTPUT_PROMPT_P10, parse_p10_output, OUTPUT_PROMPT_LIKELY, parse_likely_output
from llm_oracle.markets.base import MarketEvent


TOOL_V1_QUESTION = f"""
Predict the outcome of the following event.

{{event_text}}

Respond with:
* What you learned might know about this.
* Arguments for the event
* Arguments against the event

{OUTPUT_PROMPT_P10}
"""

TOOL_V1_PREFIX = """
You are an expert forecasting analyst with tons knowledge and skill in making calibrated conclusions. 

You never say "I don't know" and you are very thorough at using tools to perform the need research and investigation.

You always investigate both sides of an argument.

You always keep in mind that old sources might be outdated and no longer accurate.

Given the users prediction question, answer to the best of your ability and follow their instructions. 

You have access to the following tools:
"""

TOOL_V1_FORMAT_INSTRUCTIONS = """The way you use the tools is by specifying a json blob.
Specifically, this json should have a `action` key (with the name of the tool to use) and a `action_input` key (with the input to the tool going here).

The only values that should be in the "action" field are: {tool_names}

The $JSON_BLOB should only contain a SINGLE action, do NOT return a list of multiple actions. Here is an example of a valid $JSON_BLOB:

```
{{{{
  "action": $TOOL_NAME,
  "action_input": $INPUT
}}}}
```

ALWAYS use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action:
```
$JSON_BLOB
```
Observation: the result of the action
... (this Thought/Action/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question"""

TOOL_V1_SUFFIX = """
Begin! Reminder to always use the exact characters `Final Answer` when responding.
"""


class ToolAgentv1(OracleAgent):
    def __init__(
        self,
        verbose: Optional[bool] = True,
        model: Optional[llm.BaseChatModel] = None,
        tool_model: Optional[llm.BaseChatModel] = None,
        callback_manager: Optional[CallbackManager] = None,
        use_proxy: Optional[bool] = True,
    ):
        self.model = model or llm.get_default_llm()
        self.tool_model = tool_model or llm.get_default_fast_llm()
        self.verbose = verbose
        self.callback_manager = callback_manager
        self.use_proxy = use_proxy

    def get_tools(self) -> List:
        return [
            get_search_tool(),
            get_read_link_tool(summary_model=self.tool_model, use_proxy=self.use_proxy),
        ] + load_tools(["wolfram-alpha", "llm-math"], llm=self.model)

    def get_agent_kwargs(self) -> Dict:
        return {
            "prefix": TOOL_V1_PREFIX,
            "suffix": TOOL_V1_SUFFIX,
            "format_instructions": TOOL_V1_FORMAT_INSTRUCTIONS,
        }

    def get_chain_prompt(self) -> str:
        return TOOL_V1_QUESTION

    def parse_output(self, result: str) -> float:
        return parse_p10_output(result)

    def predict_event_probability(self, event: MarketEvent) -> float:
        memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        agent_chain = initialize_agent(
            self.get_tools(),
            self.model,
            agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION,
            verbose=True,
            memory=memory,
            agent_kwargs=self.get_agent_kwargs(),
            callback_manager=self.callback_manager,
        )

        event_text = event.to_text()
        if self.verbose:
            print_text(event_text + "\n", "green")

        result = agent_chain.run(input=self.get_chain_prompt().format(event_text=event_text))
        if self.verbose:
            print_text(result + "\n", "yellow")
        return self.parse_output(result)


TOOL_V2_QUESTION = f"""
Predict the outcome of the following event.

{{event_text}}

Respond with:
* What you learned about this
* Arguments for the event
* Rebuttal to your arguments for the event
* Arguments against the event
* Rebuttal to your arguments against the event

{OUTPUT_PROMPT_P10}
"""

TOOL_V2_PREFIX = """
You are an expert forecasting analyst with the ability to make well calibrated predictions about the future. 

You never say "I don't know" and you are very thorough at using tools to perform the need research and investigation.

You always investigate both sides of an argument and weights the facts by validity of your sources.

You always keep in mind that old sources might be outdated and no longer accurate.

Given the users prediction question, answer to the best of your ability and follow their instructions. 

You have access to the following tools:
"""


class ToolAgentv2(ToolAgentv1):
    def get_agent_kwargs(self) -> Dict:
        return {
            "prefix": TOOL_V2_PREFIX,
            "suffix": TOOL_V1_SUFFIX,
            "format_instructions": TOOL_V1_FORMAT_INSTRUCTIONS,
        }

    def get_chain_prompt(self) -> str:
        return TOOL_V2_QUESTION


TOOL_V3_QUESTION = f"""
Predict the outcome of the following event.

{{event_text}}

Respond with:
* What you learned about this
* Arguments for the event
* Rebuttal to your arguments for the event
* Arguments against the event
* Rebuttal to your arguments against the event

{OUTPUT_PROMPT_LIKELY}
"""


class ToolAgentv3(ToolAgentv1):
    def get_agent_kwargs(self) -> Dict:
        return {
            "prefix": TOOL_V2_PREFIX,
            "suffix": TOOL_V1_SUFFIX,
            "format_instructions": TOOL_V1_FORMAT_INSTRUCTIONS,
        }

    def get_chain_prompt(self) -> str:
        return TOOL_V3_QUESTION

    def parse_output(self, result: str) -> float:
        return parse_likely_output(result)
