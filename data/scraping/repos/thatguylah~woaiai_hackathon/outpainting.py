## This function is used as an entry point to a conversation handler in telegram bot.
## It is called when the command /outpainting is issued by the user.
## It then receives an image from the user, whilst rejecting any invalid messages (non images)
## It then stores that image in an s3 bucket in aws and returns a message to the user.
## The conversation handler then continues and prompts the user for a second image, again to be stored in s3.
## The conversation handler then calls the outpainting function, which is left to be defined for now.

import openai
import logging
from dotenv import dotenv_values
import json
import boto3
import unicodedata
from datetime import datetime
import io
from .utils import run_in_threadpool_decorator

from telegram import __version__ as TG_VER
from telegram.ext import (
    ContextTypes,
    ConversationHandler,
    CommandHandler,
    MessageHandler,
    filters,
)
from telegram import (
    Update,
    ReplyKeyboardMarkup,
    ReplyKeyboardRemove,
)

try:
    from telegram import __version_info__
except ImportError:
    __version_info__ = (0, 0, 0, 0, 0)  # type: ignore[assignment]

if __version_info__ < (20, 0, 0, "alpha", 1):
    raise RuntimeError(
        f"This example is not compatible with your current PTB version {TG_VER}. To view the "
        f"{TG_VER} version of this example, "
        f"visit https://docs.python-telegram-bot.org/en/v{TG_VER}/examples.html"
    )

# get config
config = dotenv_values(".env")
# get API tokens
HF_TOKEN = config["HF_API_KEY"]
openai.api_key = config["OPENAI_API_KEY"]

# Enable logging
logging.basicConfig(
    format="%(asctime)s - %(processName)s - %(threadName)s - [%(thread)d] - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


UPLOAD_IMAGE, PROCESS_IMAGE = range(13, 15)


async def outpainting_process_start(update: Update, context: ContextTypes):
    context.user_data["editing_image_job"] = {
        "job_type": "outpainting",
        "base_image_s3_key": None,
        "outpaint_direction": None,
    }

    buttons_lst = [["Left"], ["Right"], ["Top"], ["Bottom"]]
    output_text = "Hi! You have triggered an /outpainting workflow.\n\nWhich direction would you like to outpaint / expand your image?\nSelect an option below.\n\nSend /cancel to exit the outpainting workflow."

    # ask user to select one of the options
    await update.message.reply_html(
        f"{output_text}",
        reply_markup=ReplyKeyboardMarkup(
            buttons_lst, resize_keyboard=True, one_time_keyboard=True
        ),
    )
    return UPLOAD_IMAGE


async def outpainting_process_upload_image(update: Update, context: ContextTypes):
    selected_direction = update.message.text
    logger.log(logging.INFO, f"selected_direction: {selected_direction}")
    context.user_data["editing_image_job"][
        "outpaint_direction"
    ] = selected_direction.lower()

    await update.message.reply_text(
        "Upload the image you would like to outpaint / expand and type out what you would like the expanded regions to contain in the caption (e.g., purple skies, blue background).\n\nSend /cancel to exit the outpainting workflow."
    )

    return PROCESS_IMAGE


async def outpainting_process_terminate(update: Update, context: ContextTypes):
    del context.user_data["editing_image_job"]

    await update.message.reply_text(
        "You have terminated the outpainting workflow.\n\nPlease send /outpainting to start again or send /start for a new conversation."
    )
    return ConversationHandler.END


class ImageProcessor:
    def __init__(self) -> None:
        # Start s3 and sns clients
        self.s3_client = boto3.client("s3")
        self.sqs_client = boto3.client("sqs", region_name="ap-southeast-1")
        self.QueueUrl = QUEUE_URL = config["SQS_URL"]
        # self.base_image_s3_key = None
        # self.mask_image_s3_key = None

        self.bucket_name = BUCKET_NAME = config["BUCKET_NAME"]
        self.state = None

    @run_in_threadpool_decorator(name="aws_io")
    def upload_to_s3(self, file_stream, BUCKET_NAME, s3_key):
        response = self.s3_client.upload_fileobj(file_stream, self.bucket_name, s3_key)
        logger.log(logging.INFO, f"response: {response}")
        return 0

    @run_in_threadpool_decorator(name="aws_io")
    def put_to_sqs(self, MessageBody):
        MessageBody = json.dumps(MessageBody)

        response = self.sqs_client.send_message(
            QueueUrl=self.QueueUrl, MessageBody=MessageBody
        )
        logger.log(logging.INFO, f"response:{response}")
        return 0

    async def outpainting_process_image(self, update: Update, context: ContextTypes):
        # self.state = ConversationHandler.END

        update_as_dict = update.to_dict()
        update_as_json = json.dumps(update_as_dict)

        logger.log(logging.INFO, f"update_as_json: {update_as_json}")

        if (
            update.message.chat.username is None
        ):  ## User does not have username. @handle on tele.
            username = update.message.from_user.first_name
        else:
            username = update.message.chat.username

        # clean_username = unicodedata.name(username)
        clean_username = username

        if (
            update.message.photo
        ):  # User uploaded an image. Put the image into s3 bucket.Put update_as_json to SQS queue
            # Initialize timestamp for uniqueness and file stream buffer
            timestamp_str = datetime.now().strftime("%Y%m%d%H%M%S")
            file_stream = io.BytesIO()

            # Get file name and file id from telegram update
            file_id = update.message.photo[-1].file_id
            file_name = f"{file_id}{timestamp_str}.jpg"
            file = await update.message.photo[-1].get_file()

            # Download file to file stream buffer
            await file.download_to_memory(out=file_stream)
            file_stream.seek(0)  # Reset file stream buffer pointer to start of buffer

            s3_key = f"input/outpaint-image/{clean_username}/{file_name}"
            await self.upload_to_s3(file_stream, self.bucket_name, s3_key)
            # self.mask_image_s3_key = s3_key
            context.user_data["editing_image_job"]["base_image_s3_key"] = s3_key

            try:
                MessageBody = update_as_dict
                MessageBody["editing_image_job"] = context.user_data[
                    "editing_image_job"
                ]

                await self.put_to_sqs(MessageBody)

                await update.message.reply_text(
                    "Your image has been received!🙂 Your request is currently being processed, the image will be sent to you once it is completed.\n\nThis conversation has ended. Please send /outpainting to process a new image or send /start for a new conversation."
                )
                return ConversationHandler.END
            except Exception as e:
                logger.log(logging.ERROR, f"Exception caught here:{e}")
                await update.message.reply_text(
                    "Sorry, your job has failed to submit, please try again or contact woaiai.\n\nSend /outpainting to process a new image or /start for a new conversation."
                )
                return ConversationHandler.END

        else:
            await update.message.reply_text(
                "Please upload an image 🙂\n\nSend /cancel to stop the outpainting workflow."
            )
            return PROCESS_IMAGE


image_processor_instance = ImageProcessor()

outpainting_handler = ConversationHandler(
    entry_points=[CommandHandler("outpainting", outpainting_process_start)],
    states={
        UPLOAD_IMAGE: [
            MessageHandler(
                filters.Regex("(Left|Right|Top|Bottom)"),
                outpainting_process_upload_image,
                block=False,
            )
        ],
        PROCESS_IMAGE: [
            MessageHandler(
                filters.PHOTO,
                image_processor_instance.outpainting_process_image,
                block=False,
            )
        ],
    },
    name="OutpaintingBot",
    persistent=True,
    block=False,
    fallbacks=[
        CommandHandler("cancel", outpainting_process_terminate),
        CommandHandler("outpainting", outpainting_process_start),
    ],
)
