from langchain.agents import Tool, AgentExecutor, conversationAgent
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)

import os


# TODO: Comming with v2
class DialogAgent(Node):
    """An agent designed to hold a conversation in addition to using tools."""

    def __init__(
        self,
        get_relevant_information_callback,
        execute_action_callback,
        temperature=0.0,
        max_history_buffer_tokens=500,
    ):
        """Initialize the agent."""
        super().__init__("DialogNode")
        # Define tools. TODO: Move to folders and map properly.
        tools = [
            Tool(
                name="Search",
                func=get_relevant_information_callback,
                description="A tool that allows you to search for information from your previous experiences, \
                                useful when you need to access details from past conversations or tasks. \
                                Use a query as input to find the information you need.",
            ),
            Tool(
                name="Execute",
                func=execute_action_callback,
                description="A tool that allows you to perform physical actions, \
                                useful when you need to navigate to a location or manipulate an object. \
                                Use a predicated and object as input to perform the action.",
            ),
        ]
        self.emotion = "Happy"
        self.goal = "To have a conversation with the user."
        self.thought = "Such a beautiful day!"
        self.dialog_recommendations = "Be friendly with the user."
        conversation_prompt = self.get_conversation_prompt(tools)

        # Create conversation chain.
        self.conversation_chain = LLMChain(
            llm=ChatOpenAI(temperature=temperature),
            prompt=conversation_prompt,
            verbose=True,
        )
        self.conversation_agent = self._initialize_agent(tools)

    def update_behavioral_context(self, emotion, goal, thought):
        self.emotion = emotion
        self.goal = goal
        self.thought = thought

    def update_recommendations(self, recommendations):
        self.dialog_recommendations = recommendations

    def _initialize_agent(self, tools):
        tool_names = [tool.name for tool in tools]
        agent = conversationAgent(
            llm_chain=self.conversation_chain,
            allowed_tools=tool_names,
            ai_prefix=DEF_AI_PREFIX,
        )
        # Create agent executor.
        self.agent_executor = AgentExecutor.from_agent_and_tools(
            agent=agent, tools=tools, verbose=True
        )

    def get_conversation_prompt(self, tools, ai_prefix, user_prefix, user_suffix):
        instruction_prompt = conversationAgent.create_prompt(
            tools=tools,
            ai_prefix=ai_prefix,
            prefix=user_prefix,
            suffix=user_suffix,
            input_variables=["user_name", "chat_history"],
        )
        system_message_prompt = SystemMessagePromptTemplate(prompt=instruction_prompt)

        human_message_prompt = HumanMessagePromptTemplate.from_template(
            "New input: {input}\n\
            {agent_scratchpad}"
        )
        return ChatPromptTemplate.from_messages(
            [system_message_prompt, human_message_prompt]
        )

    def get_response(self, user_input, user_name):
        return self.agent_executor.run(input=user_input, user_name=user_name)


# DELETE?
class DialogAgent(Agent):
    """An agent designed to dialog with the user and communicate with other modules of the system."""

    output_parser: BaseOutputParser

    @property
    def _agent_type(self) -> str:
        raise NotImplementedError

    @property
    def observation_prefix(self) -> str:
        """Prefix to append the observation with."""
        return "Observation: "

    @property
    def llm_prefix(self) -> str:
        """Prefix to append the llm call with."""
        return "Thought:"

    @classmethod
    def create_prompt(
        cls,
        tools: Sequence[BaseTool],
        system_message: str = PREFIX,
        human_message: str = SUFFIX,
        input_variables: Optional[List[str]] = None,
        output_parser: Optional[BaseOutputParser] = None,
    ) -> BasePromptTemplate:
        tool_strings = "\n".join(
            [f"> {tool.name}: {tool.description}" for tool in tools]
        )
        tool_names = ", ".join([tool.name for tool in tools])
        _output_parser = output_parser or AgentOutputParser()
        format_instructions = human_message.format(
            format_instructions=_output_parser.get_format_instructions()
        )
        final_prompt = format_instructions.format(
            tool_names=tool_names, tools=tool_strings
        )
        if input_variables is None:
            input_variables = ["input", "chat_history", "agent_scratchpad"]
        messages = [
            SystemMessagePromptTemplate.from_template(system_message),
            MessagesPlaceholder(variable_name="chat_history"),
            HumanMessagePromptTemplate.from_template(final_prompt),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ]
        return ChatPromptTemplate(input_variables=input_variables, messages=messages)

    def _extract_tool_and_input(self, llm_output: str) -> Optional[Tuple[str, str]]:
        try:
            response = self.output_parser.parse(llm_output)
            return response["action"], response["action_input"]
        except Exception:
            raise ValueError(f"Could not parse LLM output: {llm_output}")

    def _construct_scratchpad(
        self, intermediate_steps: List[Tuple[AgentAction, str]]
    ) -> List[BaseMessage]:
        """Construct the scratchpad that lets the agent continue its thought process."""
        thoughts: List[BaseMessage] = []
        for action, observation in intermediate_steps:
            thoughts.append(AIMessage(content=action.log))
            human_message = HumanMessage(
                content=TEMPLATE_TOOL_RESPONSE.format(observation=observation)
            )
            thoughts.append(human_message)
        return thoughts

    @classmethod
    def from_llm_and_tools(
        cls,
        llm: BaseLanguageModel,
        tools: Sequence[BaseTool],
        callback_manager: Optional[BaseCallbackManager] = None,
        system_message: str = PREFIX,
        human_message: str = SUFFIX,
        input_variables: Optional[List[str]] = None,
        output_parser: Optional[BaseOutputParser] = None,
        **kwargs: Any,
    ) -> Agent:
        """Construct an agent from an LLM and tools."""
        cls._validate_tools(tools)
        _output_parser = output_parser or AgentOutputParser()
        prompt = cls.create_prompt(
            tools,
            system_message=system_message,
            human_message=human_message,
            input_variables=input_variables,
            output_parser=_output_parser,
        )
        llm_chain = LLMChain(
            llm=llm,
            prompt=prompt,
            callback_manager=callback_manager,
        )
        tool_names = [tool.name for tool in tools]
        return cls(
            llm_chain=llm_chain,
            allowed_tools=tool_names,
            output_parser=_output_parser,
            **kwargs,
        )
