from telegram import Update
from telegram.ext import ContextTypes
from langchain.document_loaders import OnlinePDFLoader, PyPDFLoader


async def echo(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
	# if (not hasattr(update.message.document, "mime_type") 
	#  	or update.message.document.mime_type != "application/pdf"):
	# 	await update.message.reply_text("Please load PDF")
	# 	return
	# file = await context.bot.get_file(update.message.document.file_id)
	# loader = PyPDFLoader(file.file_path)
	# pages = loader.load_and_split()
	# print(pages[0])
	
	# print(await context.bot.get_file(update.message.document.file_id))
	# print(update.message.document)
	await update.message.reply_text("No commands to handle the input!")
