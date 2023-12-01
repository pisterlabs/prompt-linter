class DialogueAgent:
    from langchain.chat_models import ChatOpenAI
    from langchain.schema import SystemMessage
    def __init__(
        self,
        name: str,
        system_message: SystemMessage,
        model: ChatOpenAI,
    ) -> None:
        self.name = name
        self.system_message = system_message
        self.model = model
        self.prefix = f"{self.name}: "
        self.reset()
    def reset(self):
        self.message_history = ["Here is the conversation so far."]
    def send(self) -> str:
        from langchain.schema import HumanMessage
        """
        Applies the chatmodel to the message history
        and returns the message string
        """
        message = self.model(
            [
                self.system_message,
                HumanMessage(content="\n".join(self.message_history + [self.prefix])),
            ]
        )
        return message.content
    def receive(self, name: str, message: str) -> None:
        """
        Concatenates {message} spoken by {name} into message history
        """
        self.message_history.append(f"{name}: {message}")


class DialogueSimulator:
    from typing import List, Callable
    def __init__(
        self,
        agents: List[DialogueAgent],
        selection_function: Callable[[int, List[DialogueAgent]], int],
    ) -> None:
        self.agents = agents
        self._step = 0
        self.select_next_speaker = selection_function
    def reset(self):
        for agent in self.agents:
            agent.reset()
    def inject(self, name: str, message: str):
        """
        Initiates the conversation with a {message} from {name}
        """
        for agent in self.agents:
            agent.receive(name, message)
        # increment time
        self._step += 1
    def step(self) -> tuple[str, str]:
        # 1. choose the next speaker
        speaker_idx = self.select_next_speaker(self._step, self.agents)
        speaker = self.agents[speaker_idx]
        # 2. next speaker sends message
        message = speaker.send()
        # 3. everyone receives message
        for receiver in self.agents:
            receiver.receive(speaker.name, message)
        # 4. increment time
        self._step += 1
        return speaker.name, message


class DialogueAgentWithTools(DialogueAgent):
    from typing import List
    from langchain.chat_models import ChatOpenAI
    from langchain.schema import SystemMessage
    def __init__(
        self,
        name: str,
        system_message: SystemMessage,
        model: ChatOpenAI,
        tool_names: List[str],
        **tool_kwargs,
    ) -> None:
        from langchain.agents import load_tools
        super().__init__(name, system_message, model)
        self.tools = load_tools(tool_names, **tool_kwargs)
    def send(self) -> str:
        from langchain.memory import ConversationBufferMemory
        from langchain.agents import initialize_agent
        from langchain.schema import AIMessage
        from langchain.agents import AgentType
        """
        Applies the chatmodel to the message history
        and returns the message string
        """
        agent_chain = initialize_agent(
            self.tools,
            self.model,
            agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
            verbose=True,
            memory=ConversationBufferMemory(
                memory_key="chat_history", return_messages=True
            ),
        )
        message = AIMessage(
            content=agent_chain.run(
                input="\n".join(
                    [self.system_message.content] + self.message_history + [self.prefix]
                )
            )
        )
        return message.content


def generate_agent_description(conversation_description, name):
    from langchain.chat_models import ChatOpenAI
    from langchain.schema import (
        HumanMessage,
        SystemMessage,
    )
    import os
    agent_descriptor_system_message = SystemMessage(
        content="You can add detail to the description of the conversation participant."
    )
    word_limit = 50  # word limit for task brainstorming
    agent_descriptor_human_message = HumanMessage(
        content=f"""{conversation_description}
        Please reply with a creative description of {name}, in {word_limit} words or less. 
        Speak directly to {name}.
        Give them a point of view.
        Do not add anything else."""
    )
    agent_specifier_prompt = [
        agent_descriptor_system_message,
        agent_descriptor_human_message,
    ]
    agent_description = ChatOpenAI(model_name=os.getenv('OPENAI_MODEL'), temperature=0)(agent_specifier_prompt).content
    return agent_description


def generate_system_message(conversation_description, name, description):
    return f"""{conversation_description}
    
Your name is {name}.

Your description is as follows: {description}

Your goal is to persuade your conversation partner of your point of view.

DO look up information with your tool to refute your partner's claims.
DO cite your sources.

DO NOT fabricate fake citations.
DO NOT cite any source that you did not look up.

Do not add anything else.

Stop speaking the moment you finish speaking from your perspective.
"""


def run_debate(_topic):
    _ans, _steps = "", ""
    from langchain.chat_models import ChatOpenAI
    from langchain.schema import (
        HumanMessage,
        SystemMessage,
    )
    names = {
        "AI accelerationist": ["arxiv", "ddg-search", "wikipedia"],
        "AI alarmist": ["arxiv", "ddg-search", "wikipedia"],
    }
    conversation_description = f"""Here is the topic of conversation: {_topic}
The participants are: {', '.join(names.keys())}"""
    agent_descriptions = {name: generate_agent_description(conversation_description, name) for name in names}
    for name, description in agent_descriptions.items():
        print(description)
    agent_system_messages = {
        name: generate_system_message(conversation_description, name, description)
        for (name, tools), description in zip(names.items(), agent_descriptions.values())
    }
    for name, system_message in agent_system_messages.items():
        print(name)
        print(system_message)
    word_limit = 50  # word limit for task brainstorming
    topic_specifier_prompt = [
        SystemMessage(content="You can make a topic more specific."),
        HumanMessage(
            content=f"""{_topic}
            You are the moderator.
            Please make the topic more specific.
            Please reply with the specified quest in {word_limit} words or less. 
            Speak directly to the participants: {*names,}.
            Do not add anything else."""
        ),
    ]
    from langchain.callbacks import get_openai_callback
    import os
    with get_openai_callback() as cb:
        specified_topic = ChatOpenAI(model_name=os.getenv('OPENAI_MODEL'), temperature=0)(topic_specifier_prompt).content
        print(f"Original topic:\n{_topic}\n")
        print(f"Detailed topic:\n{specified_topic}\n")
        # we set `top_k_results`=2 as part of the `tool_kwargs` to prevent results from overflowing the context limit
        agents = [
            DialogueAgentWithTools(
                name=name,
                system_message=SystemMessage(content=system_message),
                model=ChatOpenAI(model_name=os.getenv('OPENAI_MODEL'), temperature=0), # model_name="gpt-4"
                tool_names=tools,
                top_k_results=2,
            )
            for (name, tools), system_message in zip(
                names.items(), agent_system_messages.values()
            )
        ]
        from typing import List
        def select_next_speaker(step: int, agents: List[DialogueAgent]) -> int:
            idx = (step) % len(agents)
            return idx
        simulator = DialogueSimulator(
            agents=agents,
            selection_function=select_next_speaker
        )
        simulator.reset()
        simulator.inject("Moderator", specified_topic)
        print(f"(Moderator): {specified_topic}")
        print("\n")
        max_iters = 6
        n = 0
        while n < max_iters:
            name, message = simulator.step()
            print(f"({name}): {message}")
            print("\n")
            n += 1
        _token_cost = f"Tokens: {cb.total_tokens} = (Prompt {cb.prompt_tokens} + Completion {cb.completion_tokens}) Cost: ${format(cb.total_cost, '.5f')}"
        print(_token_cost)
        _steps = f"{_token_cost}\n\n"
    
    return [_ans, _steps]



if __name__ == "__main__":
    
    _topic = "The current impact of automation and artificial intelligence on employment"
    _re1 = run_debate(_topic)
    print(_re1)

