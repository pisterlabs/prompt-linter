import logging
from typing import List, Callable

from langchain.agents import load_tools, initialize_agent, AgentType
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage,
)
from nextcord.ext import commands

from cogs.status import working, wait_for_orders

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def setup(bot):
    bot.add_cog(DebateCog(bot, "NONE", False))


class DialogueAgent:
    def __init__(
            self,
            name: str,
            system_message: SystemMessage,
            model: ChatOpenAI,
    ) -> None:
        self.message_history = None
        self.name = name
        self.system_message = system_message
        self.model = model
        self.prefix = f"{self.name}: "
        self.reset()

    def reset(self):
        self.message_history = ["Here is the conversation so far."]

    def send(self) -> str:
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
    def __init__(
            self,
            name: str,
            system_message: SystemMessage,
            model: ChatOpenAI,
            tool_names: List[str],
            **tool_kwargs,
    ) -> None:
        super().__init__(name, system_message, model)
        self.tools = load_tools(tool_names, **tool_kwargs)

    def send(self) -> str:
        """
        Applies the chatmodel to the message history
        and returns the message string
        """
        human_and_system_message_content = "\n".join(
            [self.system_message.content] + self.message_history + [self.prefix])

        agent_chain = initialize_agent(
            self.tools,
            self.model,
            agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
            verbose=True,
            memory=ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        )
        message = AIMessage(content=agent_chain.run(
            input="\n".join([
                                self.system_message.content] + \
                            self.message_history + \
                            [self.prefix])))

        return message.content


class DebateCog(commands.Cog):
    def __init__(self, bot, topic, for_real):
        # we're not for real when we're just booting up. It takes a while. We want them to set a topic first
        if not for_real or topic == "NONE":
            return

        self.bot = bot
        self.names = {
            'AI For': [
                'arxiv',
                'ddg-search',
                'wikipedia'
            ],
            'AI Against': [
                'arxiv',
                'ddg-search',
                'wikipedia'
            ],
        }
        self.topic = topic
        self.word_limit = 50  # word limit for task brainstorming

        self.conversation_description = f"""Here is the topic of conversation: {self.topic}
           The participants are: {', '.join(self.names.keys())}"""

        self.agent_descriptor_system_message = SystemMessage(
            content="You can add detail to the description of the conversation participant.")

        self.agent_descriptions = {name: self.generate_agent_description(name) for name in self.names}
        for name, description in self.agent_descriptions.items():
            logger.info(description)

        self.agent_system_messages = {name: self.generate_system_message(name, description, tools) for
                                 (name, tools), description
                                 in zip(self.names.items(), self.agent_descriptions.values())}
        for name, system_message in self.agent_system_messages.items():
            logger.info(name)
            logger.info(system_message)

        self.topic_specifier_prompt = [
            SystemMessage(content="You can make a topic more specific."),
            HumanMessage(content=
                         f"""{self.topic}

                           You are the moderator.
                           Please make the topic more specific.
                           Please reply with the specified quest in {self.word_limit} words or less. 
                           Speak directly to the participants: {*self.names,}.
                           Do not add anything else."""
                         )
        ]
        self.specified_topic = ChatOpenAI(temperature=1.0)(self.topic_specifier_prompt).content

        logger.info(f"Original topic:\n{self.topic}\n")
        logger.info(f"Detailed topic:\n{self.specified_topic}\n")

    def generate_agent_description(self, name):
        agent_specifier_prompt = [
            self.agent_descriptor_system_message,
            HumanMessage(content=
                         f"""{self.conversation_description}
                       Please reply with a creative description of {name}, in {self.word_limit} words or less. 
                       Speak directly to {name}.
                       Give them a point of view.
                       Do not add anything else."""
                         )
        ]
        agent_description = ChatOpenAI(temperature=1.0)(agent_specifier_prompt).content
        return agent_description

    def generate_system_message(self, name, description, tools):
        return f"""{self.conversation_description}

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

    def select_next_speaker(self, step: int, agents: List[DialogueAgent]) -> int:
        idx = step % len(agents)
        return idx

    @commands.command()
    async def set_topic(self, ctx, *, topic: str):
        self.topic = topic
        self.__init__(self.bot, topic, True)
        await ctx.send(f"Debate on the topic '{self.topic}' has been started. Use the 'debate' command to continue.")

    @commands.command()
    async def debate(self, ctx):
        await working(self.bot, "Debating...")
        # we set `top_k_results`=2 as part of the `tool_kwargs` to prevent results from overflowing the context limit
        agents = [DialogueAgentWithTools(name=name,
                                         system_message=SystemMessage(content=system_message),
                                         model=ChatOpenAI(
                                             model_name='gpt-3.5-turbo',
                                             temperature=0.2),
                                         tool_names=tools,
                                         top_k_results=2,
                                         ) for (name, tools), system_message in
                  zip(self.names.items(), self.agent_system_messages.values())]
        max_iters = 6
        n = 0

        simulator = DialogueSimulator(
            agents=agents,
            selection_function=self.select_next_speaker
        )
        simulator.reset()
        simulator.inject('Moderator', self.topic)
        await ctx.send(f"(Moderator): {self.topic}")

        while n < max_iters:
            name, message = simulator.step()
            await ctx.send(f"({name}): {message}")
            n += 1
        await wait_for_orders(self.bot)