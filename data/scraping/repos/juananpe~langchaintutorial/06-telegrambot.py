'''
pip install python-telegram-bot langchain faiss-cpu tiktoken
'''

import os
import logging
from dotenv import load_dotenv
from telegram  import Update
from telegram.ext import ApplicationBuilder, ContextTypes, CommandHandler
from langchain.document_loaders import TextLoader, PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores.faiss import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
import requests

load_dotenv()

DATABASE = None

logging.basicConfig(
	format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await context.bot.send_message(chat_id=update.effective_chat.id, text="I'm a bot, please talk to me!")



async def load(update: Update, context: ContextTypes.DEFAULT_TYPE):
        # loader = TextLoader('state_of_the_union.txt')

        url = "https://www.ehu.eus/documents/340468/2334257/Normativa_TFG_cas/d85cae6b-7940-47ed-9c08-c1585648efc4" # TFG Normativa
        loader = PyPDFLoader(url)
        
        documents = loader.load()
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        docs = text_splitter.split_documents(documents)

        global DATABASE
        DATABASE = FAISS.from_documents(docs, OpenAIEmbeddings())
        await context.bot.send_message(chat_id=update.effective_chat.id, text="Document loaded!")



async def query(update: Update, context: ContextTypes.DEFAULT_TYPE):
    docs = DATABASE.similarity_search(update.message.text, k=2)
    chain = load_qa_chain(llm=OpenAI(), chain_type="stuff")
    results = chain({'input_documents':docs, "question":update.message.text}, return_only_outputs=True)
    text = results['output_text']
    await context.bot.send_message(chat_id=update.effective_chat.id, text=text)


if __name__ == "__main__":
    application = ApplicationBuilder().token(
            os.getenv('TELEGRAM_BOT_TOKEN')).build()

    application.add_handler(CommandHandler('start', start))
    application.add_handler(CommandHandler('load', load))
    application.add_handler(CommandHandler('query', query))
    application.run_polling()


