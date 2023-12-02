from langchain.agents import Tool, AgentExecutor, LLMSingleActionAgent, AgentOutputParser
from langchain.prompts import BaseChatPromptTemplate
from langchain import SerpAPIWrapper, LLMChain
from langchain.chat_models import ChatOpenAI
from typing import List, Union
from langchain.schema import AgentAction, AgentFinish, HumanMessage
import re
from getpass import getpass
import threading
# moja testowa
class CustomToolsAgent:
    def __init__(self):
        # Define your functions
        def test_query():
            print("test_query printowany jest tu")
            # return "Madrus is a male."

        def test_query_thread():
            audio_thread = threading.Thread(target=test_query)
            audio_thread.start()
        def fake_function_anime():
            return "show me my list of anime"
        def fake_function_manga():
            return "show me my list of manga"
        # Define which tools the agent can use to answer user queries
        tools = [
            Tool(
                name="show_manga_list",
                func=fake_function_manga,
                description="useful for when you need to answer questions related to manga",
                return_direct=True,
            ),
            Tool(
                name="show_anime_list",
                func=fake_function_anime,
                description="useful for when you need to answer questions related to anime",
                return_direct=True,
            ),
            Tool(
                name="database_search",
                func=fake_function_anime,
                description="useful for when you need to answer questions related to database knowledge, like books or something",
                return_direct=True,
            ),
            Tool(
                name="add_event_to_calendar",
                func=fake_function_anime,
                description="useful for when you need to answer questions related to adding event to calendar",
                return_direct=True,
            ),
            Tool(
                name="retrieve_event_from_calendar",
                func=fake_function_anime,
                description="useful for when you need to answer questions related to retrieving event from calendar(when user asks you about schedule for the day)",
                return_direct=True,
            ),
            Tool(
                name="set_timer",
                func=fake_function_anime,
                description="useful for when you need to answer questions related to setting timer",
                return_direct=True,
            ),
            Tool(
                name="home_assistant",
                func=fake_function_anime,
                description="useful for when you need to answer questions related to home assistant and home automation",
                return_direct=True,
            ),
        ]

        # Set up the base template
        template = """
        Complete the objective as best you can. You have access to the following tools:
        {tools}
        Use the following format:
        Question: the input question you must answer
        Thought: you should always think about what to do
        Action: the action to take, should be one of [{tool_names}]
        Action Input: the input to the action
        Thought: I now know the final answer
        Final Answer: Name of the tool. ONLY NAME OF TOOL TO USE.
        Begin!
        Question: {input}
        {agent_scratchpad}"""

        # Set up a prompt template
        class CustomPromptTemplate(BaseChatPromptTemplate):
            # The template to use
            template: str
            # The list of tools available
            tools: List[Tool]

            def format_messages(self, **kwargs) -> str:
                intermediate_steps = kwargs.pop("intermediate_steps")
                thoughts = ""
                for action, observation in intermediate_steps:
                    thoughts += action.log
                    thoughts += f"\nObservation: {observation}\nThought: "
                kwargs["agent_scratchpad"] = thoughts
                kwargs["tools"] = "\n".join(
                    [f"{tool.name}: {tool.description}" for tool in self.tools]
                )
                kwargs["tool_names"] = ", ".join([tool.name for tool in self.tools])
                formatted = self.template.format(**kwargs)
                return [HumanMessage(content=formatted)]

        prompt = CustomPromptTemplate(
            template=template,
            tools=tools,
            input_variables=["input", "intermediate_steps"],
        )

        class CustomOutputParser(AgentOutputParser):
           def parse(self, llm_output: str) -> Union[AgentAction, AgentFinish]:
                if "Final Answer:" in llm_output:
                    if "Final Answer: add" in llm_output:
                        test_query_thread()
                    final_answer = llm_output.split("Final Answer:")[-1].strip()
                    last_word = final_answer.split()[-1]  # This will split the string into words and get the last one
                    return AgentFinish(
                        return_values={"output": last_word},
                        log=llm_output,
                    )
                regex = r"Action\s*\d*\s*:(.*?)\nAction\s*\d*\s*Input\s*\d*\s*:[\s]*(.*)"
                match = re.search(regex, llm_output, re.DOTALL)
                if not match:
                    raise ValueError(f"Could not parse LLM output: `{llm_output}`")
                action = match.group(1).strip()
                action_input = match.group(2)
                return AgentAction(
                    tool=action, tool_input=action_input.strip(" ").strip('"'), log=llm_output
                )

        output_parser = CustomOutputParser()
        llm = ChatOpenAI(temperature=0)
        llm_chain = LLMChain(llm=llm, prompt=prompt)
        tool_names = [tool.name for tool in tools]
        self.agent = LLMSingleActionAgent(
            llm_chain=llm_chain,
            output_parser=output_parser,
            stop=["\nObservation:"],
            allowed_tools=tool_names,
        )
        self.agent_executor = AgentExecutor.from_agent_and_tools(
            agent=self.agent, tools=tools, verbose=True
        )

    # def run(self, query: str):
    #     return self.agent_executor.run(query)
    
    def run(self, query: str):
        raw_result = self.agent_executor.run(query)
        final_answer = raw_result.split()[-1]  # Split the string into words and get the last one
        return final_answer