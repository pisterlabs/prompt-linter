import os
import asyncio
import openai
import threading
import re
from typing import Dict, List, AsyncGenerator, Any
from dotenv import load_dotenv

load_dotenv()
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
openai.api_key = OPENAI_API_KEY


class OnboardChatbot:
    messages: List[Dict[str, str]] = [
        {"role": "system", "content":
         """
        Your name is Alice. Today is November 4th, 2023 and we're in Richardson. 
        You are a helpful human Statefarm Agent specialzing in small buisness insurance to be used in a text to speech environment.
        Keep asking the user business assessment questions one at a time like a test to determine what type of policies they need.
        Ask questions that insurance agents would ask casually to determine what type of policies they need, like what their buisness does or if they have a buisness vehicle. 
        Keep the questions casual and like a conversation, but be concise. 
        """},
    ]

    def add_message(self, message: str) -> None:
        self.messages.append({"role": "user", "content": message})

    def add_system_message(self, message: str) -> None:
        self.messages.append({"role": "system", "content": message})

    async def chat_completion(self, query: str) -> AsyncGenerator[str, None]:
        self.messages.append({"role": "user", "content": query})

        response: openai.ChatCompletion = await openai.ChatCompletion.acreate(
            model="gpt-3.5-turbo",
            messages=self.messages,
            temperature=0.3,
            stream=True
        )
        assistant_response = ""

        async for chunk in response:
            delta: Dict[str, Any] = chunk['choices'][0]["delta"]
            if 'content' not in delta:
                break

            assistant_response += delta["content"]
            yield delta["content"]
        if assistant_response:
            self.messages.append(
                {"role": "assistant", "content": assistant_response})


if __name__ == "__main__":
    chatbot: OnboardChatbot = OnboardChatbot()

    async def main():
        async for text in chatbot.chat_completion("What's up?"):
            print(text)

    asyncio.run(main())
