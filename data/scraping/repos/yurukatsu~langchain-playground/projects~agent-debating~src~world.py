from typing import Dict, List

from langchain.chat_models.base import BaseChatModel
from langchain.output_parsers import OutputFixingParser, PydanticOutputParser
from langchain.prompts import (
    HumanMessagePromptTemplate,
    PromptTemplate,
    SystemMessagePromptTemplate,
)
from langchain.schema import SystemMessage
from langchain.tools import BaseTool
from pydantic.v1 import BaseModel, Field

from .agents import DialogueAgentWithTools
from .prompts.agents import AGENT_SYSTEM_MESSAGE_TEMPLATE
from .prompts.moderator import TOPIC_SPECIFIER_PROMPT_TEMPLATE
from .prompts.participants import CREATE_PARTICIPANT_PROMPT_TEMPLATE


class Participant(BaseModel):
    """
    Participant

    name (str): Participant name.
    personality (str): Participant personality.
    objective (str): Participant objective in conversation.
    """

    name: str = Field(description="participant name")
    personality: str = Field(description="participant's personality")
    objective: str = Field(description="participant's objective in conversation")


class ParticipantWithTools(Participant):
    """
    Participant

    name (str): Participant name.
    personality (str): Participant personality.
    objective (str): Participant objective in conversation.
    model (BaseChatModel | None, optional): LLM.
    available_tools (List[BaseTool] | None, optional):
        Available tools of the participant.
    """

    model: BaseChatModel | None = Field(default=None, description="LLM")
    available_tools: List[BaseTool] | None = Field(
        default=None, description="available tools"
    )


class ParticipantList(BaseModel):
    """
    Participant list

    participants (List[Participant]): Participants.
    """

    participants: List[Participant] = Field(description="participant list")


class ParticipantGenerator:
    """
    Class to generate participants
    """

    @classmethod
    def create_participants(
        cls,
        llm: BaseChatModel,
        situation: str,
        n_participants: int = 2,
        template: str = CREATE_PARTICIPANT_PROMPT_TEMPLATE,
    ) -> List[Participant]:
        """
        Create participants in some situation

        Args:
            llm (BaseChatModel): LLM.
            situation (str): Situation.
            n_participants (int, optional): Number of participants. Defaults to 2.
            template (str, optional): Template. Defaults to CREATE_PARTICIPANT_PROMPT_TEMPLATE.

        Returns:
            List[Participant]: Participants.
        """

        output_parser = PydanticOutputParser(pydantic_object=ParticipantList)
        output_fixing_parser = OutputFixingParser.from_llm(
            parser=output_parser, llm=llm
        )

        prompt = PromptTemplate(
            input_variables=["situation", "n_participants"],
            partial_variables={
                "format_instructions": output_parser.get_format_instructions()
            },
            template=template,
        )

        input_variables = {
            "situation": situation,
            "n_participants": n_participants,
        }

        text = prompt.format(**input_variables)

        completion = llm.predict(text=text)
        participants = output_fixing_parser.parse(completion).participants

        return participants

    @classmethod
    def create_participants_for_debate(
        cls,
        llm: BaseChatModel,
        topic: str,
        n_participants: int = 2,
        template: str = CREATE_PARTICIPANT_PROMPT_TEMPLATE,
    ) -> List[Participant]:
        """
        Create participants in a debate situation

        Args:
            llm (BaseChatModel): LLM.
            topic (str): Debate topic.
            n_participants (int, optional): Number of participants. Defaults to 2.
            template (str, optional): Template. Defaults to CREATE_PARTICIPANT_PROMPT_TEMPLATE.

        Returns:
            List[Participant]: Participants.
        """

        situation = f"""Debates will be held on the following topics.
        Topic: {topic}

        Generate information about the participants.
        """
        return cls.create_participants(
            llm, situation, n_participants, template=template
        )


class Environment(BaseModel):
    """
    Environment

    topic (str): Conversation topic.
    model (BaseChatModel | None, optional): LLM.
    tools (List[BaseTool] | None, optional): Tools.
    """

    topic: str = Field()
    model: BaseChatModel = Field()
    tools: List[BaseTool] | None = Field()


class World:
    def __init__(
        self,
        environment: Environment,
        participants: List[Participant | ParticipantWithTools],
        **agent_kwargs,
    ) -> None:
        """
        World

        Args:
            environment (Environment): Environment
            participants (List[Participant]): Participants
        """
        self.environment = environment
        self.participants = []
        for participant in participants:
            if isinstance(participant, Participant):
                self.participants.append(ParticipantWithTools(**participant.dict()))
            elif isinstance(participant, ParticipantWithTools):
                self.participants.append(participant)
            else:
                raise TypeError
        self.agents = self.create_agents(**agent_kwargs)

    @property
    def participant_list(self) -> List[str]:
        """
        Participant name list.

        Returns:
            List[str]: Participant name list
        """
        return [participant.name for participant in self.participants]

    @property
    def conversation_description(self) -> str:
        """
        Conversation Description
        """

        return f"""Here is the topic of conversation: {self.environment.topic}
The participants are: {', '.join(self.participant_list)}"""

    def create_agents(self, **agent_kwargs) -> List[DialogueAgentWithTools]:
        """
        Create agents

        Returns:
            List[DialogueAgentWithTools]: Agents.
        """

        def _create_agent(participant: Participant) -> DialogueAgentWithTools:
            """
            Create agent

            Args:
                participant (Participant): Participant

            Returns:
                DialogueAgentWithTools: Agent
            """
            template_kwargs = dict(
                conversation_description=self.conversation_description,
                name=participant.name,
                personality=participant.personality,
                objective=participant.objective,
            )
            system_message = SystemMessagePromptTemplate.from_template(
                AGENT_SYSTEM_MESSAGE_TEMPLATE
            ).format(**template_kwargs)

            model = (
                self.environment.model
                if participant.model is None
                else self.environment.model
            )

            tools = (
                self.environment.tools
                if participant.available_tools is None
                else participant.available_tools
            )

            agent = DialogueAgentWithTools(
                participant.name,
                system_message,
                model,
                tools,
                **agent_kwargs,
            )
            return agent

        # Create all agents
        agents = []
        for participant in self.participants:
            agent = _create_agent(participant)
            agents.append(agent)

        return agents


class DebateWorld(World):
    def __init__(
        self,
        environment: Environment,
        participants: List[Participant | ParticipantWithTools],
        **agent_kwargs,
    ) -> None:
        """
        Debate World

        Args:
            environment (Environment): Environment
            participants (List[Participant]): Participants
        """
        super().__init__(environment, participants, **agent_kwargs)

    def concretize_topic(self, llm: BaseChatModel | None = None, word_limit: int = 100):
        template_kwargs = dict(
            topic=self.environment.topic,
            word_limit=word_limit,
            names=", ".join(self.participant_list),
        )
        topic_specifier_prompt = [
            SystemMessage(content="You can make a topic more specific."),
            HumanMessagePromptTemplate.from_template(
                TOPIC_SPECIFIER_PROMPT_TEMPLATE
            ).format(**template_kwargs),
        ]
        if llm is None:
            llm = self.environment.model
        return llm(topic_specifier_prompt).content
