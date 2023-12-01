import base64
import time
import traceback
from bot import WhatsApp
import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from enum import Enum
from db import User, Message
import datetime
import json
from environment import get_env
from openai import AsyncOpenAI
import httpx
from tools import internet_search_with_bing
from io import BytesIO




class FlowHandler():
    def __init__(self, settings: dict):
        self.messenger = WhatsApp(os.getenv("TOKEN"), phone_number_id=os.getenv("PHONE_NUMBER_ID"))
        engine = create_engine('postgresql://postgress:mysecretpassword@localhost:5432/test')
        Session = sessionmaker(bind=engine)
        self.session = Session()
        self.TYPES = Enum("TYPES",["ask", "image", "friend"])
        self.settings = settings
        self.gpt = AsyncOpenAI(api_key=get_env("OPENAI_API_KEY"))



    def handle_request(self, data: dict):
        phone_number = self.messenger.get_mobile(data)
        message = self.messenger.get_message(data)
        message_type = self.get_message_type(message)
        user = self.session.query(User).filter("phone_number" == phone_number).first()
        if user is None:
            self.handle_initialize_request(phone_number, message_type)
            return
        
        if message_type == self.TYPES.ask:
            self.handle_ask_initialize_message(phone_number)
        elif message_type == self.TYPES.image:
            self.handle_image_intialize_message(phone_number)
        elif message_type == self.TYPES.friend:
            self.handle_friend_initialize_request(data)
        else:
            self.handle_request_message(message, phone_number)


    def send_message(self, data: dict):
        name = self.messenger.get_name(data)
        mobile = self.messenger.get_mobile(data)
        self.messenger.send_message(f"Hi {name}, nice to connect with you", mobile)
    
    def handle_initialize_request(self, phone_number: str, message_type):
        new_user = User(
            phone_number=phone_number,
            # message_type='ask',
            # username='username',
            # chat_id='chat_id',
            country_code='country_code',
            phone='phone',
            # language_code='language_code',
            total_messages=0,
            last_activity=datetime.datetime.now(),
            created_at=datetime.datetime.now(),
            # utm_content='utm_content',
            # utm_source='utm_source',
            # utm_medium='utm_medium',
            # utm_campaign='utm_campaign',
            message_limit=self.settings["message_limit"],
            image_limit=self.settings["image_limit"],
            total_image_generation=0
        )
        self.session.add(new_user)
        self.session.commit()
        if message_type in None:
            pass
            # send initial message  
    

    def get_message_type(self, message: str):
        if message in self.TYPES:
            return message
        return None
    
    def handle_ask_initialize_message(self, phone_number: str):
        user = self.session.query(User).filter("phone_number" == phone_number).first()
        if user is None:
            return
        user.message_type = self.TYPES.ask
        res = self.messenger.send_message("What do you want to ask?", phone_number)
        if res.status_code == 200:
            self.session.commit()
    
    def handle_image_initialize_message(self, phone_number: str):
        user = self.session.query(User).filter("phone_number" == phone_number).first()
        if user is None:
            return
        user.message_type = self.TYPES.image
        res = self.messenger.send_message("What do you want to draw?", phone_number)
        if res.status_code == 200:
            self.session.commit()

    def handle_friend_initialize_message(self, phone_number: str):
        user = self.session.query(User).filter("phone_number" == phone_number).first()
        if user is None:
            return
        user.message_type = self.TYPES.friend
        res = self.messenger.send_message("What do you want to draw?", phone_number)
        if res.status_code == 200:
            self.session.commit()

    def handle_request_message(self, message: str, phone_number: str):
        user = self.session.query(User).filter("phone_number" == phone_number).first()
        if user is None:
            return
        if user.message_limit <= user.total_messages:
            self.handle_limit_reached_message(message, phone_number)
            return
        if user.message_type == self.TYPES.ask:
            self.handle_ask_message(message, phone_number)
        elif user.message_type == self.TYPES.image:
            self.handle_image_message(message, phone_number)
        elif user.message_type == self.TYPES.friend:
            self.handle_friend_message(message, phone_number)
        else:
            self.handle_default_message(message, phone_number)
    
    def handle_default_message(self, message: str, phone_number: str):
        self.messenger.send_message("I don't understand", phone_number)

    def handle_limit_reached_message(self, message: str, phone_number: str):
        self.messenger.send_message("You have reached your message limit", phone_number)


    async def handle_ask_message(self, message: str, phone_number: str):
        try:
            last_messages = self.session.query(Message).filter("user_phone_number" == phone_number).order_by("created_at").limit(self.settings["message_history_lenght"])
            messages = [
                    {
                        "role": "user",
                        "content": self.settings.get("ask_command_system")
                    }
                ]

            for m in last_messages:
                # this messages not contains intro or command message
                messages.append(json.loads(m.get("message")))
            
            messages.append({
                "role": "user",
                "content": message
            })
            st = time.time()
            try:
                gpt_response = await self.gpt.chat.completions.create(
                    messages=messages,
                    model=self.settings.get("gpt_model"),
                    temperature=self.settings.get("gpt_temperature"),
                    max_tokens=self.settings.get("gpt_max_tokens"),
                    timeout=self.settings.get("chatgpt_chat_timeout"),
                    tools=[
                        {
                            "type": "function",
                            "function": {
                                "name": "internet_search",
                                "description": "Search the internet for an answer to your question. Like weather, news, concerts, etc.",
                                "parameters": {
                                    "type": "object",
                                    "properties": {
                                        "search_query": {
                                            "type": "string",
                                            "description": "The query to search for. Create optimized search query question for asked question.",
                                        },
                                    },
                                    "required": ["search_query"],
                                },
                            }
                            
                        }
                    ],
                    tool_choice="auto",
                )
            except httpx.ReadTimeout:
                self.messenger.send_message(self.settings["chatgpt_timeout_message"], phone_number)
                # raise httpx.ReadTimeout("Gpt Timeout Error")

            if gpt_response.choices[0].finish_reason == "tool_calls":
                if gpt_response.choices[0].message.tool_calls[0].function.name == "internet_search":
                    call_message = gpt_response.choices[0].message.model_dump()
                    if "function_call" in call_message:
                        del call_message["function_call"]
                    messages.append(call_message)
                    search_query = json.loads(gpt_response.choices[0].message.tool_calls[0].function.arguments)["search_query"]
                    try:
                        search_result = await internet_search_with_bing(search_query)
                    except Exception as e:
                        self.messenger.send_message(self.settings["chatgpt_timeout_message"], phone_number)

                    messages.append({
                        "role": "tool",
                        "tool_call_id": gpt_response.choices[0].message.tool_calls[0].id,
                        "content": search_result,
                    })
                    new_message = Message(
                        user_phone_number='1234567890',
                        content=search_result,
                        message_type='ask',
                        created_at=datetime.datetime.now()
                    )

                    self.session.add(new_message)
                    self.session.commit()
                    try:
                        gpt_response = await self.gpt.chat.completions.create(
                        messages=messages,
                        model=self.settings.get("gpt_model"),
                        temperature=self.settings.get("gpt_temperature"),
                        max_tokens=self.settings.get("gpt_max_tokens"),
                        timeout=self.settings.get("chatgpt_chat_timeout"),
                        )
                    except httpx.ReadTimeout:
                        self.messenger.send_message(self.settings["chatgpt_timeout_message"], phone_number)
                        httpx.ReadTimeout("Gpt Timeout Error")
                    self.session(User).filter("phone_number" == phone_number).first().total_messages += 1
                    self.session.commit()
            # calculate elapsed time
            et = time.time() - st
            # return response to user
            # log assistant message
            new_message = Message(
                user_phone_number='1234567890',
                content=gpt_response.choices[0].message.content,
                message_type='ask',
                created_at=datetime.datetime.now()
            )

            self.session.add(new_message)
            self.session.commit()
            self.messenger.send_message(gpt_response.choices[0].message.content, phone_number)
            user = self.session(User).filter("phone_number" == phone_number).first()
            user.total_messages += 1
            user.last_activity = datetime.datetime.now()
            self.session.commit()

        except Exception as e:
            print(f"Error happened. Error: {e}, traceback: {traceback.format_exc()}")

    async def handle_image_message(self, message: str, phone_number: str):
        try:
            new_message = Message(
                user_phone_number='1234567890',
                content=message,
                message_type='image',
                created_at=datetime.datetime.now()
            )
            self.session.add(new_message)
            user = self.session.query(User).filter("phone_number" == phone_number).first()
            user.last_activity = datetime.datetime.now()
            self.session.commit()

            if user.image_limit <= user.total_image_generation:
                self.messenger.send_message(self.settings["rate_limit_message"], phone_number)
                return

            await self.messenger.send_message(self.settings["image_wait_message"], phone_number)

            st = time.time()
            try:
                response = await self.gpt.images.generate(
                    model="dall-e-3",
                    prompt=message,
                    size="1024x1024",
                    quality="standard",
                    n=1,
                    response_format="b64_json",
                    timeout=self.settings.get("chatgpt_image_timeout")
                    )
            except httpx.ReadTimeout:
                await self.messenger.send_message(self.settings["chatgpt_timeout_message"], phone_number) 
                raise httpx.ReadTimeout
            
            # calculate elapsed time
            et = time.time() - st
            image = BytesIO(base64.b64decode(response.data[0].b64_json))

            # return response to user
            new_message = Message(
                user_phone_number=phone_number,
                content=response.model_dump(),
                message_type='image',
                created_at=datetime.datetime.now()
            )
            self.session.add(new_message)
            user = self.session.query(User).filter("phone_number" == phone_number).first()
            user.total_image_generation += 1
            self.session.commit()
            self.messenger.send_image(image, phone_number)
            # log assistant message
            
        except Exception as e:
            print(f"Error happened. Error: {e}, traceback: {traceback.format_exc()}")

    async def handle_friend_message(self, message: str, phone_number: str):
        pass






"""
handle request 

        check user message_type

        or if it is initialize message 

        send appropriate answers

        

    hande ask request 

        genreate using chatgpt and send message

    handle image request 

        generate image and send message

    hanlde friend request
        i dont know what to do with this

"""