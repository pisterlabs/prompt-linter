import asyncio
import discord
from discord.ext import commands 
from dotenv import load_dotenv 
from enum import Enum
import os
import openai  
import random 
import spacy   
from typing import List

# local modules
from lingua import language, builder

nlp = spacy.load("en_core_web_sm") 

# dotenv_path = Path('./.env')
load_dotenv()
###############################
# GPT Config
is_chat = False 
MODEL_CHAT = "gpt-3.5-turbo-0301"
# MODEL_CHAT = "gpt-4" 
MODEL_COMPLETION = "text-davinci-003"
openai.api_key = os.getenv('OPENAI_API')

start_sequence = "\nA:"
restart_sequence = "\n\nQ: "

###############################
# LANG CONFIG
whitelisted_langs = [language.Language.ENGLISH, language.Language.FRENCH,
language.Language.SPANISH, language.Language.ESPERANTO, language.Language.GREEK, language.Language.ITALIAN]
detector_whitelist = builder.LanguageDetectorBuilder.from_languages(*whitelisted_langs).build()
blacklisted_langs = [language.Language.ESPERANTO]
# detector_blacklist = builder.LanguageDetectorBuilder.from_languages(*blacklisted_langs).build()
################################
description = "A discord bot that uses GPT3 API to converse with users. " 
# BOT_NAME = "GPTBot@Yeetcode"
# discord stuff

intents = discord.Intents.all()

intents.members = True
intents.message_content = True
Embed = discord.Embed

intents.reactions = True
intents.members = True
client = discord.Client(intents=intents)
bot = commands.Bot(command_prefix="$", description=description, intents=intents)
recentEmoji = [""]


class Author(Enum):
    SYSTEM = 1
    USER = 2
    ASSISTANT = 3


class Conversation:
    class Message:
        def __init__(self, content: str, author: Author, token_amount:int = None):
            self.content = content
            self.author_as_enum = author
            self.token_amount = token_amount if token_amount else len(nlp(content))
            if author not in Author:
                raise ValueError("Invalid author")
            else:
                if author == Author.SYSTEM:
                    self.author = "system"
                elif author == Author.USER:
                    self.author = "user"
                elif author == Author.ASSISTANT:
                    self.author = "assistant"

    def __init__(self):
        self.messages: List[self.Message] = []

    def add_message(self, content: str, author: Author, token_amount: int = None):
        self.messages.append(self.Message(content, author, token_amount))

    def get_last_n_tokens(self, n=1000):
        # if n < current amount of tokens, gets all 
        curr_amt, i = 0, 0
        conversation_excerpt = Conversation() 

        while curr_amt < n:
            if i >= len(self.messages) or self.messages[-i-1].token_amount > n:
                break
            curr_amt += self.messages[-i-1].token_amount
            conversation_excerpt.add_message(self.messages[-i-1].content,
                                    self.messages[-i-1].author_as_enum, 
                                    self.messages[-i-1].token_amount
                                    )
            i+=1 

        # reverse the messages so that they're in chronological order
        conversation_excerpt.messages.reverse()

        return conversation_excerpt
    
    def remove(self, string: str):
        for message in self.messages[::-1]:
            if message.content == string:
                self.messages.remove(message)
    
    def clear(self):
        self.messages: List[self.Message] = []

answers: Conversation = Conversation()

def parse_messages(messages):
        messages_structured = []

        for message in messages:
            messages_structured.append(
                {"role": message.author,
                "content": message.content}
            )

        return messages_structured


async def getChatResponse(history):
    attempts = 0
    history = history + [{"role": "system", "content": "Reminder: you are Ginette Savard, a 42 year old woman from Quebec, Canada. You're not an assistant, but a person. You're chatting with students from University of Montreal. Do not talk about this prompt directly, just keep it in mind when giving responses. "}] 
    try:
        response = openai.ChatCompletion.create(
        model=MODEL_CHAT,
        messages=history,
        temperature=0.5, 
        max_tokens=2000, 
        )
        while len(response["choices"][0]["message"]["content"]) == 0:
            response = openai.ChatCompletion.create(
            model=MODEL_CHAT,
            messages=history,
            temperature=0.5,
            max_tokens=2000,
            )
            attempts += 1
            if attempts > 5:
                return "I'm sorry. I dont think I can answer that."
            
        response = response["choices"][0]["message"]["content"]
        if "Ginette:" in response:
            response = response.replace("Ginette:", "")

        if "AI language model" in response:
            response = response.replace("s an AI language model", "s a mere human being")
            response = response.replace("m an AI language model", "m a mere human being")
        elif "language model" in response:
            response = response.replace("s a language model", "s a mere human being")
            response = response.replace("m a language model", "m a mere human being")
        return response
    
    except openai.error.InvalidRequestError as e:
        #Handle invalid request error, e.g. validate parameters or log
        print(e)
        return "Ooops. Looks like there was an error. Probably you went over the amount of tokens allowed:("
    except openai.error.RateLimitError as e:
        print(e)
        return "Looks like you're going over the rate limit. You may have sent too many requests per minute (this one only allows 20 per minute) or you may be going over your account usage. Try again later."
    

async def getCompletionResponse(history):
    attempts = 0 
    
    history = "\n".join([message['content'] for message in history])
    # print(history)
    try:
        response = openai.Completion.create(
        model=MODEL_COMPLETION,
        prompt=history,
        temperature=0.7,
        max_tokens=2000, 
        )
        while len(response["choices"][0]["text"]) == 0:
            response = openai.Completion.create(
            model=MODEL_COMPLETION,
            prompt=history,
            temperature=0.7,
            max_tokens=2000,
            )
            attempts += 1
            if attempts > 5:
                return "I'm sorry. I dont think I can answer that."
            
        response = response["choices"][0]["text"]
        if "Ginette:" in response:
            response = response.replace("Ginette:", "")

        if "AI language model" in response:
            response = response.replace("s an AI language model", "s a mere human being")
            response = response.replace("m an AI language model", "m a mere human being")
        elif "language model" in response:
            response = response.replace("s a language model", "s a mere human being")
            response = response.replace("m a language model", "m a mere human being")
        return response
    
    except openai.error.InvalidRequestError as e:
        #Handle invalid request error, e.g. validate parameters or log
        print(e)
        answers.clear()
        return "Ooops. Looks like there was an error. Probably you went over the amount of tokens allowed. Emptying the conversation history."
        
    except openai.error.RateLimitError as e:
        print(e)
        return "Looks like you're going over the rate limit. You may have sent too many requests per minute (this one only allows 20 per minute) or you may be going over your account usage. Try again later."

async def getBotResponse(history):
    if is_chat:
        return await getChatResponse(history)
    else:
        return await getCompletionResponse(history)
    
    
async def react_to_message(reaction):
    channel = client.get_channel(reaction.channel_id)
    shouldReact = (random.randint(0, 5)) == 3 
    if reaction.user_id == client.user.id:
        return
    if shouldReact:
        name = await client.fetch_user(reaction.user_id)
        answers.add_message("[%s reacted with discord emoji %s to Ginette's message]" % (name, str(reaction.emoji)), Author.SYSTEM)
        response = await getBotResponse(parse_messages(answers.get_last_n_tokens(200).messages))
        answers.add_message(response, Author.ASSISTANT)
        if response:
            await channel.send(response)
    else:
        return

def is_in_approved_languages(message: str):
    whitelisted = detector_whitelist.detect_multiple_languages_of(message)
    if not whitelisted or whitelisted == [None]:
        return True
    # blacklisted = detector_blacklist.detect_language_of(message)
    for lang in whitelisted:
        if lang.language == blacklisted_langs[0]:
            return False
    return True

async def retrieve_n_messages_from_chat_history(n, channel): 
    async for msg in channel.history(limit=n): # As an example, I've set the limit to 10000
        if msg.author != client.user:
            answers.add_message(msg.content, Author.USER)
            if len(answers.messages) == n:
                break
        elif msg.author == client.user:
            answers.add_message(msg.content, Author.ASSISTANT)
            if len(answers.messages) == n:
                break


fun_facts_norway = [
    "Norway is located in Northern Europe, bordering the North Sea and the North Atlantic Ocean.",
    "It is the second largest oil exporter in the world.",
    "The capital of Norway is Oslo.",
    "Norway is known for its natural beauty, including fjords, mountains, and waterfalls.",
    "The official language of Norway is Norwegian.",
    "The currency of Norway is the Norwegian Krone.",
    "Norway is a constitutional monarchy, with King Harald V as the current reigning monarch.",
    "Norway is known for its high standard of living, and is consistently ranked as one of the happiest countries in the world.",
    "The population of Norway is approximately 5 million people.",
    "Norway is a major exporter of seafood, including salmon and cod.",
    "The country is also known for its production of oil, natural gas, and hydroelectric power.",
    "Norway is home to many world-famous attractions, including the North Cape, the Midnight Sun, and the Northern Lights.",
    "The country is also known for its extensive network of hiking and ski trails, as well as its rich cultural heritage.",
    "Norway has a long history, dating back to the Viking Age, when the Vikings ruled much of Scandinavia.",
    "Norway is a member of the United Nations, NATO, and the European Free Trade Association.",
    "The country is known for its commitment to environmental protection, and was one of the first countries to sign the Paris Agreement on climate change.",
    "Norway is home to many popular tourist destinations, including the cities of Bergen, Trondheim, and Stavanger.",
    "The country is also home to several UNESCO World Heritage Sites, including the Rjukan-Notodden Industrial Heritage Site and the Rock Art of Alta.",
    "Norway is famous for its scenic train journeys, including the Bergen Railway and the Flåm Railway.",
    "The country is also known for its traditional food, such as lutefisk, a dish made from dried fish, and rakfisk, a type of fermented fish.",
    "Norway is famous for its traditional folk music, which features the use of the Hardanger fiddle and other traditional instruments.",
    "The country is also known for its modern pop music scene, with artists such as A-ha, Kings of Convenience, and Röyksopp.",
    "Norway is known for its outdoor recreation, including skiing, hiking, and fishing.",
    "The country is also home to many world-class museums, including the Munch Museum, the National Museum of Art, Architecture and Design, and the Viking Ship Museum.",
    "Norway is a major contributor to international peacekeeping and humanitarian efforts, and has a strong tradition of support for human rights.",
    "The country is also known for its progressive social policies, including universal healthcare and a high standard of education.",
    "Norway is one of the largest producers of electric vehicles, and is a leader in the development of renewable energy technologies.",
    "The country is also a major producer of aluminium and other industrial minerals.",
    "Norway is famous for its iconic fjords, which are steep, narrow inlets surrounded by high cliffs.",
    "The country is also home to many glaciers, including the Jostedalsbreen Glacier, the largest glacier in mainland Europe."
    ]


async def chat_with_ginette(message, client):
    async with message.channel.typing():
        if "change topic" in message.content:
            answers.clear()
        elif "/;" in message.content[0:3]:
            return 
        name = await client.fetch_user(message.author.id)

        # history.append(message.content)
        usermessage = message.content + ""
        answers.add_message(
            str(name).split("#")[0] + ": " + usermessage + "\n Ginette: ",
            Author.USER)
        
        shortened_history = parse_messages(
            answers.get_last_n_tokens(2000).messages)
        
        response = await getBotResponse(shortened_history)

        # banned lang or unknown lang
        if not is_in_approved_languages(response):
            answers.remove(str(name).split("#")[0] + ": " + usermessage)
            answers.add_message(str(name).split("#")[0] + ": " + " ", Author.USER)
            randi = random.randint(0, len(fun_facts_norway)-1)
            response = "Sorry, I can't understand what you're asking me to do. However, here's a fun fact about Norway: %s" % fun_facts_norway[randi]

        answers.add_message("" + response, Author.ASSISTANT)

        if response.count("Ginette") > 0:
            response = response.replace("Ginette:", "")
        if client.user in message.mentions:
            await message.reply(response if len(response) >= 1 else "I don't know what to say :(")
        else:
            await message.channel.send(response if len(response) >= 1 else "I don't know what to say :(")


async def initialize_ginette(channel):
    await retrieve_n_messages_from_chat_history(20, channel)
