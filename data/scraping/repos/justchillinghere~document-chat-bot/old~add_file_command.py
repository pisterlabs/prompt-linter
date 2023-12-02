from telegram.ext import CommandHandler, ContextTypes
from telegram import File, Update, Document
import typing
from langchain.document_loaders import PyPDFLoader
from error_handler import logger

from PDF_handlerlangchain import Dialog

async def add_file(update: Update, context: ContextTypes.DEFAULT_TYPE) -> File | None:
	logger.info("Checking file format...")
	if (not hasattr(update.message.document, "mime_type") 
	 	or update.message.document.mime_type != "application/pdf"):
		await update.message.reply_text("Please load PDF")
		return
	logger.info("Started loading file")
	file = await context.bot.get_file(update.message.document.file_id)
	await context.bot.send_message(
        chat_id=update.effective_chat.id, text="Please wait for the file to be uploaded"
    )
	Dialog(user_id=update.message.from_user.id).load_document_to_vec_db(
		file_name=update.message.document.file_name,
		file_path=file.file_path
		)
	await context.bot.send_message(
        chat_id=update.effective_chat.id, text="File has been uploaded successfully!"
    )