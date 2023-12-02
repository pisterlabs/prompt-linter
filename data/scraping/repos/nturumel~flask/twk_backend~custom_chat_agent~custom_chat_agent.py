from typing import Dict, List

from langchain import LLMChain, OpenAI
from langchain.agents import AgentExecutor, LLMSingleActionAgent, Tool
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationEntityMemory

from twk_backend.custom_chat_agent.custom_output_parser import CustomOutputParser
from twk_backend.custom_chat_agent.custom_prompt import CustomPromptTemplate
from twk_backend.custom_chat_agent.example_refine_chain import ExampleRefineChain


class CustomChatAgent:
    def __init__(
        self,
        tools: List[Tool],
        name: str,
        description: str,
        temperature: float,
        samples: List[Dict],
        personality: str,
    ):
        prompt = CustomPromptTemplate(
            tools=tools,
            name=name,
            personality=personality,
            description=description,
            # This omits the `agent_scratchpad`, `tools`, and `tool_names`
            # variables because those are generated dynamically
            # This includes the `intermediate_steps` variable because that is needed
            input_variables=[
                "input",
                "intermediate_steps",
                "history",
                # "agent_scratchpad",
                "entities",
            ],
        )
        output_parser = CustomOutputParser()
        llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=temperature)

        # LLM chain consisting of the LLM and a prompt
        llm_chain = LLMChain(llm=llm, prompt=prompt)

        tool_names = [tool.name for tool in tools]
        agent = LLMSingleActionAgent(
            llm_chain=llm_chain,
            output_parser=output_parser,
            stop=["\nObservation:"],
            allowed_tools=tool_names,
        )

        memory = ConversationEntityMemory(llm=OpenAI())
        self.agent_executor = AgentExecutor.from_agent_and_tools(
            agent=agent, tools=tools, verbose=True, memory=memory
        )

        if samples:
            self.refine_chain = ExampleRefineChain(samples)
        else:
            self.refine_chain = None

    def chat(self, user_input: str):
        response = self.agent_executor.run(user_input)
        if self.refine_chain:
            response = self.refine_chain.refine_response(
                input=user_input, response=response
            )

        return response
