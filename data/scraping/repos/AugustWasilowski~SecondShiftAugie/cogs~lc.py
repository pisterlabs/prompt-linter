import logging
import os
import pickle
import textwrap

from langchain.chains import RetrievalQA, RetrievalQAWithSourcesChain
from langchain.chains.summarize import load_summarize_chain
from langchain.chat_models.openai import ChatOpenAI
from langchain.document_loaders import TextLoader, YoutubeLoader, UnstructuredURLLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain.vectorstores import FAISS
from nextcord.ext import commands
from pytube import YouTube

from cogs.ssa import generate_voice_sample, SSAWrapper
from cogs.status import working, wait_for_orders

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)
augie = SSAWrapper()


def setup(bot: commands.Bot):
    bot.add_cog(LangChainCog(bot))


async def exe_selfreflect(ctx, arg):
    llm = ChatOpenAI(model='gpt-3.5-turbo', openai_api_key=os.getenv("OPENAI_API_KEY"))
    embeddings = OpenAIEmbeddings(disallowed_special=(), openai_api_key=os.getenv("OPENAI_API_KEY"))

    root_dir = './cogs/'
    docs = []

    for dirpath, dirnames, filenames in os.walk(root_dir):
        # Go through each file
        for file in filenames:
            if file.endswith(".py"):  # Check if file has .py extension
                try:
                    # Load up the file as a doc and split
                    loader = TextLoader(os.path.join(dirpath, file), encoding='utf-8')
                    docs.extend(loader.load_and_split())
                except Exception as e:
                    logger.error(f"error loading docs {e}")

    docs.extend(TextLoader(os.path.join("main.py")).load_and_split())
    logger.info(f"You have {len(docs)} documents\n")
    docsearch = FAISS.from_documents(docs, embeddings)
    # Get our retriever ready
    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=docsearch.as_retriever())

    output = qa.run(arg)
    if augie.use_voice:
        await augie.speak(output, ctx, True)
    output_chunks = textwrap.wrap(output, width=2000)

    # send each chunk separately using ctx.send()
    for chunk in output_chunks:
        await ctx.send(chunk)


async def summarize_youtube_id(ctx, arg):
    loader = YoutubeLoader.from_youtube_url(arg)
    result = loader.load()  # Loads the video

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=0)
    texts = text_splitter.split_documents(result)

    llm = OpenAI(temperature=0, openai_api_key=os.getenv("OPENAI_API_KEY"))
    chain = load_summarize_chain(llm, chain_type="map_reduce", verbose=True)
    result = chain.run(texts[:4])
    generate_voice_sample(result, True)
    await ctx.send(result)


class LangChainCog(commands.Cog):
    def __init__(self, bot: commands.Bot) -> None:
        self.bot = bot
        self.is_busy = False

    @commands.command()
    async def youtube(self, ctx, *, arg):
        # Summary for SHORT YouTube videos. Must be the video ID, not the whole URL. i.e. "g8LlwlCU0EA" not
        # "https://youtu.be/g8LlwlCU0EA"
        yt = YouTube(arg)
        await summarize_youtube_id(ctx, yt.video_id)

    @commands.command()
    async def selfreflect(self, ctx, *, arg):
        await working(self.bot)
        await exe_selfreflect(ctx, arg)
        await wait_for_orders(self.bot)

    @commands.command()
    async def websites(self, ctx, *, args):
        await working(self.bot)
        urls = args.split(",")
        loaders = UnstructuredURLLoader(urls=urls, strategy="fast")
        data = loaders.load()
        text_splitter = CharacterTextSplitter(separator='\n',
                                              chunk_size=1000,
                                              chunk_overlap=200)

        docs = text_splitter.split_documents(data)
        await ctx.send(f"Num Docs: {len(docs)}")
        if not docs:
            await ctx.send("No documents found.")
            return

        embeddings = OpenAIEmbeddings()
        vectorStore_openAI = FAISS.from_documents(docs, embeddings)

        with open("faiss_store_openai.pkl", "wb") as f:
            pickle.dump(vectorStore_openAI, f)
        await wait_for_orders(self.bot)

    @commands.command()
    async def interrogate(self, ctx, *, args):
        with open("faiss_store_openai.pkl", "rb") as f:
            VectorStore = pickle.load(f)

        llm = OpenAI(temperature=0, model_name='')
        chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=VectorStore.as_retriever())
        prompt = f"Write a Title for the transcript that is under 15 words. " \
                 f"Then write: '--Summary--' " \
                 f"Write 'Summary' as a Heading " \
                 f"1. Write a summary of the provided transcript. " \
                 f"Then  write: '--Additional Info--'. " \
                 f"Then return a list of the main points in the provided transcript. " \
                 f"Then return a list of action items. " \
                 f"Then return a list of follow up questions. " \
                 f"Then return a list of potential arguments against the transcript." \
                 f"For each list, return a Heading 2 before writing the list items. " \
                 f"Limit each list item to 200 words, and return no more than 20  points per list. " \
                 f"Transcript: "
        foo = chain({"question": prompt}, return_only_outputs=True)
        await ctx.send(foo)
        await wait_for_orders(self.bot)
