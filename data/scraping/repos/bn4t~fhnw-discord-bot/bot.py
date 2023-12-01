import discord
from dotenv import load_dotenv

from langchain.llms import LlamaCpp
from langchain import PromptTemplate, LLMChain
from langchain.document_loaders import TextLoader, PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import LlamaCppEmbeddings
from langchain.vectorstores import OpenSearchVectorSearch
from langchain.docstore.document import Document
from langchain.chains.question_answering import load_qa_chain
import os

load_dotenv()  # load all the variables from the env file
bot = discord.Bot()


@bot.event
async def on_ready():
    print(f"{bot.user} is ready and online!")
    await bot.change_presence(activity=discord.Game(name="world domination!"))


@bot.slash_command(name="llm-info", description="Get info about the llm bot")
async def info(ctx):
    await ctx.respond("I am running " + os.getenv("MODEL_NAME"))


@bot.slash_command(name="ask", description="Ask the LLM something")
async def ask(ctx: discord.ApplicationContext, question: str):
    await ctx.defer()

    chain = load_qa_chain(llm, chain_type="stuff")

    docs = db.similarity_search(question)
    if len(docs) == 0:
        await ctx.respond("Sorry, I don't know the answer")

    responses = []
    for rdoc in docs:
        responses.append(Document(page_content=chain.run(input_documents=[rdoc], question=question),  metadata=rdoc.metadata))

    chain = load_qa_chain(llm, chain_type="stuff")

    while len(responses) != 1:
        c = 0
        length_context = 0
        for i in range(len(responses)):
            count_tokens = 40 + len(question.split(' ')) + length_context + len(responses[i].page_content.split(' '))
            if max_num_of_tokens > count_tokens:
                length_context += len(responses[i].page_content.split(' '))
                c += 1

        responses.append(Document(page_content=chain.run(input_documents=responses[:c], question=question)))
        responses = responses[c:]

    await ctx.respond(responses[0].page_content)


if __name__ == '__main__':
    loader = PyPDFLoader("handbuch-studierende.pdf")
    documents = loader.load()

    text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=100, separator='\n')
    split_docs = text_splitter.split_documents(documents)

    print(len(split_docs))
    if len(split_docs) > 20:
        raise "Document-is-to-big.txt"

    max_num_of_tokens = 2048
    model = "./llama-2-13b-german-assistant-v4.Q3_K_S.gguf"
    llm = LlamaCpp(model_path=model, n_ctx=max_num_of_tokens, n_threads=4)
    llm_embeddings = LlamaCppEmbeddings(model_path=model)

    persist_directory = 'db'
    db = None

    if os.path.isdir(persist_directory):
        db = Chroma(persist_directory=persist_directory, embedding_function=llm_embeddings)
    else:
        db = Chroma.from_documents(split_docs, llm_embeddings, persist_directory=persist_directory)
        db.persist()

    bot.run(os.getenv('TOKEN'))  # run the bot with the token
