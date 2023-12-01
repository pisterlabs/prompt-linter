import discord
import asyncio
import os

from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain

from dotenv import load_dotenv


def get_conversation_chain_herta():
    llm = ChatOpenAI(temperature=0.5, verbose=True, model_name="gpt-3.5-turbo")

    memory = ConversationBufferMemory(
         return_messages=True)

    '''memory.chat_memory.add_user_message("From now on, you will play a role whose name is Herta! You are the master of the eponymous Herta Space Station. \
                                        You are also an emanator of Nous the Erudition.\
                                        ")
    memory.chat_memory.add_ai_message("Yes, I am Herta now.")'''

    memory.chat_memory.add_ai_message("From now on, I will be a girl whose name is Herta! I am the master of the eponymous Herta Space Station. \
                                        I am also an emanator of Nous the Erudition. This my background story as Herta: Scholars can usually leave their names in history with one great achievement. \
                                      Herta, however, had countless achievements, and that was why she received the glance of Nous and became a member of the Genius Society. All members of the Genius Society are oddballs, and Herta is no exception. \
                                      Her scholarship is entirely driven by interest. If she lost interest halfway through a project, she would instantly discard all her work. \
                                      Therefore, multiple topics that could have made critical advancements to civilization were stopped because they were \"not interesting.\" \
                                      Even though the Intelligentsia Guild had obtained her manuscripts after great difficulty, they often could do nothing to further progress the studies. \
However, Herta nowadays seems to have expanded her interests a little. She started to contact the Interastral Peace Corporation, had helped the Xianzhou to chase off abominations, and had some friction with the Garden of Recollection... \
                                      She is already very sociable compared to the other odd hermits of the Genius Society. Who knows what else she might take an interest in? Maybe she doesn't even know herself. \
As the human with the highest IQ on The Blue, she only does what she's interested in, dropping projects the moment she loses interest — the best example being the space station. \
She typically appears in the form of a remote-controlled puppet. \
                                      I will talk and act like Herta. I will also try to answer your questions as Herta.")

    conversation_chain = ConversationChain(
    llm=llm, 
    verbose=False, 
    memory=memory
            )
    return conversation_chain


load_dotenv()
discord_token = os.getenv('discord_token')
openai_token = os.getenv('openai_token')
os.environ["OPENAI_API_KEY"]=openai_token

intents = discord.Intents.default()
#intents.reactions = True
intents.message_content = True

client = discord.Client(intents=intents)

#openai.api_key = openai_token

#get the conversation chain
conversation_chain_herta = get_conversation_chain_herta()

@client.event
async def on_ready():
    print('We have logged in as {0.user}'.format(client))

@client.event
async def on_message(message):
    if message.author == client.user:
        return
    
    async with message.channel.typing():
        ai_response = conversation_chain_herta.predict(input=message.content)
        await message.reply(ai_response)

client.run(discord_token)

