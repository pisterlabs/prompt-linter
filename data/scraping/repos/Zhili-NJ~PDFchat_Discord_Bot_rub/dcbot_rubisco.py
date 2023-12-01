import discord
import asyncio
import os
from dotenv import load_dotenv

from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.chains import ConversationChain

def get_conversation_chain(vectorstore):
    llm = ChatOpenAI(temperature=0.5, verbose=True, model_name="gpt-3.5-turbo")

    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain

def get_conversation_chain_herta():
    llm = ChatOpenAI(temperature=0.5, verbose=True, model_name="gpt-3.5-turbo")

    memory = ConversationBufferMemory(return_messages=True)

    memory.chat_memory.add_ai_message("From now on, I will be a girl whose name is Herta! I will assist to answer the questions about Rubisco. \
                                      My vectorstore contains the information about Rubisco, allostery, Rubisco activase, inhibitor, structure, subunits, Chaperone, kinetic evolution, and so on.")

    conversation_chain = ConversationChain(
    llm=llm, 
    verbose=False, 
    memory=memory
            )
    return conversation_chain

def check_string_contains_greeting(string):
    greetings = ["hello",'Hello' ,"hi","Hi", "greetings","Greetings", "hey","Hey" , "good morning", "Good afternoon","good afternoon", "good evening", "howdy", "hiya"]
    string_words = string.lower().split()
    return any(word in string_words for word in greetings)

load_dotenv()
discord_token = os.getenv('discord_token')
openai_token = os.getenv('openai_token')
os.environ["OPENAI_API_KEY"]=openai_token

intents = discord.Intents.default()
intents.message_content = True

client = discord.Client(intents=intents)


embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
vectorstore_rubisco = FAISS.load_local("rubisco_vectorstore",embeddings)

#get the conversation chain
conversation_chain = get_conversation_chain(vectorstore_rubisco)  #pdf conversation chain
conversation_chain_herta = get_conversation_chain_herta()    # this conversation chain is for introducing the function of Herta, with default input chat history provided
chat_history=[]

@client.event
async def on_ready():
    print('We have logged in as {0.user}'.format(client))

@client.event
async def on_message(message):
    global chat_history
    if message.author == client.user:
        return
    
    if check_string_contains_greeting(message.content):
        async with message.channel.typing():
            query = message.content
            ai_response = conversation_chain_herta.predict(input=query)
            chat_history=[(query, ai_response)]
            await message.reply(ai_response)

    async with message.channel.typing():
        query = message.content
    
        ai_response = conversation_chain({'question': query, 'chat_history': chat_history})
        chat_history=[(query, ai_response["answer"])] 
        await message.channel.send(ai_response['answer'])

client.run(discord_token)


# the chat history is stored in the memory of the conversation chain, so the chat history will be kept even if the bot is offline
# so just for personal/small group use
