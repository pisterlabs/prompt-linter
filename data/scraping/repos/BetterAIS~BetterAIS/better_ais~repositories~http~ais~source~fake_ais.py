"""
MIT License

Copyright (c) 2023 Illia Chaban

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
import asyncio
import datetime
import openai
import random

from better_ais.config.openai import OpenAiSettings
from .base_client import AISClient
from openai.error import RateLimitError

class FakeAISClient(AISClient):
    """FakeAisClient is a fake implementation of Client for accademic information systems."""

    def __init__(self, openai_settings: OpenAiSettings):
        self.__openai_settings = openai_settings

    async def __create_completion(self, prompt: str, max_tokens: int = 5):
        base_prompt = "Imagine you are logged in to the accademic information system of your university. You'r university is called 'STU'. Website of your accademic information system is 'https://is.stuba.sk'. "

        openai.api_key = self.__openai_settings.api_key
        try:
            response = await asyncio.to_thread(
                    openai.Completion.create, 
                    
                    engine=self.__openai_settings.model,
                    prompt=base_prompt + prompt,
                    temperature=0.8,
                    max_tokens=max_tokens
            )
        except RateLimitError:
            return "Not enough tokens. This is a fake AIS."
        return response.choices[0].text # type: ignore

    async def __gen_email(self, username: str):
        base_prompt = "You'r username is '" + username + "'. You have a new email. "
        
        return {
            'id': random.randint(0, 1000000),
            'sender': await self.__create_completion(prompt=base_prompt + "Who is the sender of the mail?", max_tokens=5),
            'subject': await self.__create_completion(prompt=base_prompt + "What is the subject of the mail?", max_tokens=5),
            'body': await self.__create_completion(prompt=base_prompt + "What is the body of the mail?", max_tokens=100),
            'is_read': False,
            'created_at': datetime.datetime.now(),
            'updated_at': datetime.datetime.now(),
        }

    async def __gen_document(self, username: str):
        base_prompt = "You'r username is '" + username + "'. You have a new document. "
        
        return {
            'id': random.randint(0, 1000000),
            'author': await self.__create_completion(prompt=base_prompt + "Who is the author of the document?", max_tokens=5),
            'title': await self.__create_completion(prompt=base_prompt + "What is the title of the document?", max_tokens=5),
            'subject': await self.__create_completion(prompt=base_prompt + "What is the subject of the document?", max_tokens=5),
            'description': await self.__create_completion(prompt=base_prompt + "What is the description of the document?", max_tokens=100),
            'link': await self.__create_completion(prompt=base_prompt + "What is the link of the document?", max_tokens=5),
            'file_path': await self.__create_completion(prompt=base_prompt + "What is the file path of the document?", max_tokens=5),
            'created_at': datetime.datetime.now(),
            'updated_at': datetime.datetime.now(),
        }
    
    async def __gen_homework(self, username: str):
        base_prompt = "You'r username is '" + username + "'. You have a new homework. "
        
        return {
            'id': random.randint(0, 1000000),
            'title': await self.__create_completion(prompt=base_prompt + "What is the title of the homework?", max_tokens=5),
            'description': await self.__create_completion(prompt=base_prompt + "What is the description of the homework?", max_tokens=100),
            'link': await self.__create_completion(prompt=base_prompt + "What is the link of the homework?", max_tokens=5),
            'created_at': datetime.datetime.now(),
            'updated_at': datetime.datetime.now(),
        }

    async def get_new_mails(self, username: str, password: str):
        # if random.randint(0, 5):
        #     return []
        
        new_mails = []
        for _ in range(random.randint(1, 5)):
            new_mails.append(await self.__gen_email(username))
    
        return new_mails

    async def get_new_documents(self, username: str, password: str):
        # if random.randint(0, 5):
        #     return []
        
        new_documents = []
        for _ in range(random.randint(1, 5)):
            new_documents.append(await self.__gen_document(username))
    
        return new_documents

    async def get_new_homeworks(self, username: str, password: str):
        # if random.randint(0, 5):
        #     return []
        
        new_homeworks = []
        for _ in range(random.randint(1, 5)):
            new_homeworks.append(await self.__gen_homework(username))
    
        return new_homeworks

    async def get_time_table(self, username: str, password: str):
        mock = [
            {
                "day": 0,
                "lesson": "1. hodina",
                "time": 8,
                "teacher": "Ing. Jonathan Joestar, PhD.",
                "room": "D2",
            }, 
            {
                "day": 0,
                "lesson": "2. hodina",
                "time": 2,
                "teacher": "Ing. George Joestar, PhD.",
                "room": "D2",
            },
            {
                "day": 1,
                "lesson": "1. hodina",
                "time": 1,
                "teacher": "Ing. Jonathan Joestar, PhD.",
                "room": "D2",
            },
            {
                "day": 1,
                "lesson": "2. hodina",
                "time": 3,
                "teacher": "Ing. George Joestar, PhD.",
                "room": "D2",
            },
            {
                "day": 2,
                "lesson": "1. hodina",
                "time": 1,
                "teacher": "Ing. Jonathan Joestar, PhD.",
                "room": "D2",
            }
        ]
        
        return mock