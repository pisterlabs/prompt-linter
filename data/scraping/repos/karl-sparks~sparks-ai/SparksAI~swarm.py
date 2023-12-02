from langchain.agents import AgentType, Tool, initialize_agent, AgentExecutor
from langchain.agents.openai_assistant import OpenAIAssistantRunnable
from langchain.callbacks.streaming_stdout_final_only import (
    FinalStreamingStdOutCallbackHandler,
)
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import MessagesPlaceholder, PromptTemplate
from langchain.utilities import SerpAPIWrapper

from SparksAI.config import MODEL_NAME


class Swarm:
    def __init__(self) -> None:
        self.conversation_swarm = {}
        self.archivist_swarm = {}
        self.analyst_swarm = self.init_analyst_agent()

        self.llm = ChatOpenAI(
            model=MODEL_NAME,
            streaming=True,
            callbacks=[FinalStreamingStdOutCallbackHandler()],
        )

    def get_conversation_agent(self, username: str) -> AgentExecutor:
        if username in self.conversation_swarm:
            return self.conversation_swarm[username]

        else:
            self.init_conversation_agent(username)

            return self.conversation_swarm[username]

    def init_conversation_agent(self, username: str) -> None:
        if username in self.conversation_swarm:
            return None

        prompt = PromptTemplate.from_template(
            """You are an expert conversationalist tasked with crafting a response to a specific question.
            An analyst has already reviewed the question and supplied guidance along with additional information to assist you.
            Furthermore, you have access to context from prior interactions with the user, ensuring your response is well-informed and tailored to the user's needs and history of inquiries.

            Analyst Review:
            {analyst_message}
            
            
            Summary of prior interactions:
            {prior_messages}


            Question:
            {input_message}
            """
        )

        convo_agent = prompt | self.llm

        self.conversation_swarm[username] = convo_agent

    def get_analyst_agent(self) -> AgentExecutor:
        return self.analyst_swarm

    def init_analyst_agent(self) -> AgentExecutor:
        search = SerpAPIWrapper()

        tools = [
            Tool(
                name="Search",
                func=search.run,
                description="Useful when you need to answer questions about current events. You should ask targeted questions.",
            ),
        ]

        ai_assistant = OpenAIAssistantRunnable.create_assistant(
            name="Conversation Analyst",
            instructions="""
            You are an analyst responsible for reviewing and responding to user messages.
            Your primary task is to evaluate each message for clarity and content, providing structured guidance on formulating comprehensive and logical responses.
            In addition to offering response strategies, you are also tasked with analyzing the messages to anticipate and formulate what a more detailed or specific follow-up question from the user might entail.
            This includes identifying any underlying assumptions, implicit queries, or additional information that the user might require for a complete understanding.
            If needed, you can provide additional information using your search function.""",
            tools=tools,
            model=MODEL_NAME,
            as_agent=True,
        )

        return AgentExecutor(agent=ai_assistant, tools=tools)

    def get_archivist(self, username: str) -> AgentExecutor:
        if username in self.archivist_swarm:
            return self.archivist_swarm[username]

        else:
            self.init_archivist(username)

            return self.archivist_swarm[username]

    def init_archivist(self, username: str) -> None:
        if username in self.archivist_swarm:
            return None

        prompt = PromptTemplate.from_template(
            """You are an archivist.
            You have been given the transcript of a conversation between an AI and a human user.
            You have also received the most recent message from the human.
            Your job is to provide a list of at least three bullet points summarizing the transcript.
            The list should contain the most relevant material to the most recent message.

            Transcript:
            {memory}

            Most recent message:
            {input_message}
            """
        )

        archivist = LLMChain(
            llm=self.llm,
            prompt=prompt,
            verbose=True,
        )

        self.archivist_swarm[username] = archivist
