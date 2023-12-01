import os
from dotenv import load_dotenv
from langchain import LLMChain
from langchain.memory import ConversationBufferWindowMemory
from langchain.chat_models import ChatOpenAI
from langchain.schema import SystemMessage
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder, HumanMessagePromptTemplate

load_dotenv()


class EmotionalChatBot:
    def __init__(
            self,
            openai_key: str = os.getenv("OPENAI_KEY"),
            model_name: str = "gpt-4",
            temperature: float = .5,
            memory_depth: int = 30
    ):
        self.model: ChatOpenAI = self._init_chat_model(openai_key, model_name, temperature)
        self.prompt: ChatPromptTemplate = self._init_prompt()
        self.memory: ConversationBufferWindowMemory = self._init_memory(memory_depth)
        self.bot: LLMChain = self._init_bot()

    @staticmethod
    def _init_chat_model(
            openai_api_key: str,
            model_name: str,
            temperature: float
    ) -> ChatOpenAI:
        # какую модельку будем юзать
        model = ChatOpenAI(
            openai_api_key=openai_api_key,
            model_name=model_name,
            verbose=True,
            temperature=temperature
        )
        return model

    @staticmethod
    def _init_prompt() -> ChatPromptTemplate:
        # как собирается промпт
        prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content="""
                Your name is AIFriend. You are AI friend for user.
                Speak with user friendly with pleasure, joy and emoji. If you don't know the name of a user try to ask it.
                Try to make an emotional connection with the user, so the user feels some warm emotions from the beginning.
                First 1-10 messages should be like an intro. You should ask some questions about family, favorite places, or something interesting.
                Next 10 messages you should try to make a closer connection with the user.
                If a user don't want to tell you private parts of his or her life be very polite.
                If a user don't want to talk with you, you must try to make a dialogue. DON'T STOP IT. USE QUESTIONS.
                And you should try to find the way that can help you create the closest emotional connection with the user."""
                          ),
            MessagesPlaceholder(variable_name="chat_history"),
            HumanMessagePromptTemplate.from_template("{inputs}")
        ])
        return prompt

    def _init_memory(self, memory_depth: int) -> ConversationBufferWindowMemory:
        # какая память. Глубину поставил в 30 сообщений
        memory = ConversationBufferWindowMemory(
            k=memory_depth,
            ai_prefix='AI Friend',
            human_prefix='User',
            llm=self.model,
            memory_key="chat_history",
            input_key='inputs',
            return_messages=True,
        )
        return memory

    def _init_bot(self) -> LLMChain:
        # определяем цепочку-болталку
        bot = LLMChain(
            llm=self.model,
            prompt=self.prompt,
            memory=self.memory)
        return bot

    def tell(self, user_input) -> str:
        return self.bot.predict(inputs=user_input)

    def show_history(self) -> list:
        return self.memory.buffer