import asyncio

from dotenv import load_dotenv

from chatbot.ai.assistants.paper_chatter.paper_chatter_prompt import PAPER_CHATTER_SYSTEM_TEMPLATE

load_dotenv()
from langchain import LLMChain, OpenAI
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationSummaryBufferMemory, VectorStoreRetrieverMemory, CombinedMemory
from langchain.prompts import (
    HumanMessagePromptTemplate,
    ChatPromptTemplate, SystemMessagePromptTemplate,
)


class PaperChatter:
    def __init__(self,
                 chat_llm,
                 prompt,
                 memory,):
        self._chat_llm = chat_llm
        self._prompt = prompt
        self._memory = memory
        self._chain = self._create_llm_chain()

    @classmethod
    async def create(cls,
                     temperature=0.8,
                     model_name="gpt-4",
                     prompt_template=PAPER_CHATTER_SYSTEM_TEMPLATE):
        chat_llm = ChatOpenAI(
            streaming=True,
            callbacks=[StreamingStdOutCallbackHandler()],
            temperature=temperature,
            model_name=model_name,
        )

        prompt = cls._create_prompt(prompt_template=prompt_template)
        memory = await cls._configure_memory(cls)

        return cls(chat_llm, prompt, memory)

    async def _configure_memory(self):
        conversation_memory = self._configure_conversation_memory(self)
        vectorstore_memory = await self._configure_vectorstore_memory(self)
        combined_memory = CombinedMemory(memories=[conversation_memory,
                                                   vectorstore_memory])
        return combined_memory

    async def _configure_vectorstore_memory(self, ):


        chroma_vector_store = await self._create_vector_store()

        retriever = chroma_vector_store.as_retriever(search_kwargs=dict(k=3))

        return VectorStoreRetrieverMemory(retriever=retriever,
                                          memory_key="vectorstore_memory",
                                          input_key="human_input",)



    def _configure_conversation_memory(self):
        return ConversationSummaryBufferMemory(memory_key="chat_memory",
                                               input_key="human_input",
                                               llm=OpenAI(temperature=0),
                                               max_token_limit=1000)

    def _create_llm_chain(self):
        return LLMChain(llm=self._chat_llm,
                        prompt=self._prompt,
                        memory=self._memory,
                        verbose=True,
                        )

    def _create_prompt(prompt_template: str):
        system_message_prompt = SystemMessagePromptTemplate.from_template(
            prompt_template
        )

        human_template = "{human_input}"
        human_message_prompt = HumanMessagePromptTemplate.from_template(
            human_template
        )

        chat_prompt = ChatPromptTemplate.from_messages(
            [system_message_prompt, human_message_prompt]
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
    async def main():
        paper_chatter = await PaperChatter.create()
        await paper_chatter.demo()


    asyncio.run(main())