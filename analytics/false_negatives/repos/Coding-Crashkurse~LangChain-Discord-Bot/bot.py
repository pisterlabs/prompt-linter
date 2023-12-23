import discord
from discord.ext import commands
from dotenv import load_dotenv, find_dotenv
import os
from langchain.prompts import SystemMessagePromptTemplate, PromptTemplate
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import TextLoader
from langchain.schema import HumanMessage

load_dotenv(find_dotenv())

loader = TextLoader("./youtube.txt")
documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=300, chunk_overlap=100)
texts = text_splitter.split_documents(documents)

embeddings = OpenAIEmbeddings()
retriever = Chroma.from_documents(texts, embeddings).as_retriever()
chat = ChatOpenAI(temperature=0)

prompt_template = """You are a helpful dicord bot that helps users with programming and answers about the channel.

{context}

Please provide the most suitable response for the users question.
Answer:"""

prompt = PromptTemplate(
    template=prompt_template, input_variables=["context"]
)
system_message_prompt = SystemMessagePromptTemplate(prompt=prompt)


intents = discord.Intents.default()
intents.message_content = True

bot = commands.Bot(command_prefix='!', intents=intents)

@bot.command()
async def question(ctx, *, question):
    try:
        docs = retriever.get_relevant_documents(query=question)
        formatted_prompt = system_message_prompt.format(context=docs)

        messages = [formatted_prompt, HumanMessage(content=question)]
        result = chat(messages)
        await ctx.send(result.content)
    except Exception as e:
        print(f"Error occurred: {e}")
        await ctx.send("Sorry, I was unable to process your question.")


bot.run(os.environ.get("DISCORD_TOKEN"))
