import asyncio

from dotenv import load_dotenv

from chatbot.ai.assistants.video_chatter.prompts.video_chatter_prompt import VIDEO_CHATTER_SYSTEM_TEMPLATE

load_dotenv()
from langchain import LLMChain, OpenAI
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationSummaryBufferMemory
from langchain.prompts import (
    HumanMessagePromptTemplate,
    ChatPromptTemplate, SystemMessagePromptTemplate,
)


class VideoChatter:
    def __init__(self,
                 temperature=0.8,
                 model_name="gpt-4",
                 prompt: str = VIDEO_CHATTER_SYSTEM_TEMPLATE,
                 ):
        self._chat_llm = ChatOpenAI(
            streaming=True,
            callbacks=[StreamingStdOutCallbackHandler()],
            temperature=temperature,
            model_name=model_name,
        )

        self._prompt = self._create_prompt(prompt_template=prompt)
        self._memory = self._configure_memory()

        self._chain = self._create_llm_chain()


    def _configure_memory(self):

        return ConversationSummaryBufferMemory(memory_key="chat_history",
                                               llm=OpenAI(temperature=0),
                                               max_token_limit=1000)

    def _create_llm_chain(self):
        return LLMChain(llm=self._chat_llm,
                        prompt=self._prompt,
                        memory=self._memory,
                        verbose=True,
                        )

    def _create_prompt(self, prompt_template: str):
        self._system_message_prompt = SystemMessagePromptTemplate.from_template(
            prompt_template
        )

        human_template = "{human_input}"
        human_message_prompt = HumanMessagePromptTemplate.from_template(
            human_template
        )

        chat_prompt = ChatPromptTemplate.from_messages(
            [self._system_message_prompt, human_message_prompt]
        )

        return chat_prompt

    async def async_process_input(self, input_text):
        print(f"Input: {input_text}")
        print("Streaming response...\n")
        ai_response = await self._chain.arun(human_input=input_text)
        return ai_response

    async def demo(self):
        print("Welcome to the Neural Control Assistant demo!")
        print("Type 'exit' to end the demo.\n")

        while True:
            input_text = input("Enter your input: ")

            if input_text.strip().lower() == "exit":
                print("Ending the demo. Goodbye!")
                break

            response = await self.async_process_input(input_text)

            print("\n")

    async def load_memory_from_thread(self, thread, bot_name: str):
        async for message in thread.history(limit=None, oldest_first=True):
            if message.content == "":
                continue
            if str(message.author) == bot_name:
                self._memory.chat_memory.add_ai_message(message.content)
            else:
                self._memory.chat_memory.add_user_message(message.content)


if __name__ == "__main__":
    assistant = VideoChatter()
    asyncio.run(assistant.demo())
