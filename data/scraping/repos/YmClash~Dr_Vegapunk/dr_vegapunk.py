import discord
import openai
import os
import quotes
import random
import logging
from dotenv import load_dotenv
from langchain.llms import OpenAI
from langchain.agents import load_tools, initialize_agent, AgentType
from langchain.tools import Tool
from langchain.utilities import GoogleSearchAPIWrapper
from decimal import Decimal,getcontext
# from substrateinterface import SubstrateInterface
load_dotenv()
#
# rpc_point = SubstrateInterface(url="wss://rpc-parachain.baju.network")

key = os.getenv('DR_VEGAPUNK_API_KEY')
openai.api_key = os.getenv('OPENAI_API_KEY')
GOOGLE_CSE_ID = os.getenv('GOOGLE_CSE_ID')
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')

# on  reunisalise  tout   les  module


intents = discord.Intents.default()
intents.message_content = True

bot = discord.Client(intents=intents)

search = GoogleSearchAPIWrapper()

lili = OpenAI(temperature=0.9)

tool = load_tools(["arxiv"])

tools = [Tool.from_function(
    func=search.run,
    name="Search",
    description="usefu  for answer  about current events"
),
]

agent_chain = initialize_agent(tools,
                               lili,
                               agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
                               verbose=True)


async def demande_balance(adresse):
    # rpc_point = SubstrateInterface(url="wss://rpc-parachain.baju.network")
    getcontext().prec = 15
    result = rpc_point.query('System','Account',[adresse])
    balance = Decimal(result.value['data']['free']) / Decimal(10**12)
    balance_format = format(balance,'.2f')
    return f'la Balance est de compte est de  : {balance_format} Bajun '


async def demande_longchain(prompt) :
    agent_chain.run(prompt)


# on va configure l'Api  de  OpenAi   enfin  d'avoir  acce au  Model de ChatGPT  et aussi a Dall-e

# Model  ChatGPT3.5
async def demande_gpt(prompt) :
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role" : "system", "content" : "Je suis le Dr Vegapunk"},
            {"role" : "user", "content" : prompt}
        ],
        max_tokens=500,
        temperature=0.75,
        top_p=1.0,
        # stop =4,
        frequency_penalty=0.0,
        presence_penalty=0.6
    )

    message = response.choices[0].message.content.strip()
    return message


# Model Dalle

async def demande_image(prompt) :
    response = openai.Image.create(
        prompt=prompt,
        n=1,
        size="1024x1024"
    )
    image_url = response['data'][0]['url']
    return image_url


@bot.event
async def on_ready() :
    print(f'We have logged in #momo-jam-cava as : {bot.user} , SUPER!!!!!!!!!!!!!!!!!  dev by Y_mC ')


@bot.event
async def on_message(message) :
    if message.author == bot.user :
        return

    if message.content.startswith('roll 10') :
        roll = [random.randint(1, 100) for i in range(10)]
        await message.channel.send(f'{roll}')

    if message.content.startswith('quote') :
        quote = random.choice(quotes.Afro_quote_1)
        await message.channel.send(f'{quote}')

    dr_vegapunk_channel: discord.TextChannel = bot.get_channel(1099103151572926477)
    if message.content.startswith("!") :
        prompt = message.content[11 :]
        response = await demande_gpt(prompt)
        await dr_vegapunk_channel.send(content=response)

    if message.content.startswith("!img") :
        prompt = message.content[11 :]
        response = await demande_image(prompt)
        await dr_vegapunk_channel.send(content=response)

    if message.content.startswith("!archiv") :
        prompt = message.content[11 :]
        response = await demande_longchain(prompt)
        await dr_vegapunk_channel.send(content=response)

    if message.content.startswith('!baj'):
        adresse = message.content[11 :]
        response = await demande_balance(adresse)
        await dr_vegapunk_channel.send(content=response)



bot.run(key, log_level=logging.DEBUG)

# generator = pipeline('text-generation', model='gpt2-xl')
# tokenizer = GPT2Tokenizer.from_pretrained('gpt2-xl')
# tokenizer.pad_token = tokenizer.eos_token
# dr_vegapunk = GPT2LMHeadModel.from_pretrained('gpt2-xl')
#
#
# def generate_response(prompt, model, tokenizer, max_length=50) :
#     inputs = tokenizer(prompt,
#                        return_tensors="pt",
#                        padding=True,
#                        truncation=True,
#                        max_length=max_length)
#     input_ids = inputs["input_ids"]
#     attention_mask = inputs["attention_mask"]
#     output = model.generate(input_ids,
#                             max_length=max_length,
#                             num_return_sequences=1,
#                             no_repeat_ngram_size=2,
#                             do_sample=True,
#                             temperature=0.1,
#                             attention_mask=attention_mask,
#                             pad_token_id=tokenizer.eos_token_id)
#     response = tokenizer.decode(output[0], skip_special_tokens=True)
#     print(response)
#
#     # return response
#
#
# while True :
#     prompt = input(f'YMC: ')
#     generate_response(prompt, dr_vegapunk, tokenizer)
