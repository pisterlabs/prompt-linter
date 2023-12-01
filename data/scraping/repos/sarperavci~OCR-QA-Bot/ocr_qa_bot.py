import os
import telegram
from telegram.ext import Updater, MessageHandler, Filters,CommandHandler
import requests
import openai

# Set up your OCR.space API key
API_KEY = 'K81753893888957' #OCR.space API.
openai.api_key = "sk-"  #OpenAI Key
TOKEN =  '' #Telegram Bot TOKEN



def create(text):
    response = openai.ChatCompletion.create(
      model="gpt-3.5-turbo",
      messages=[

              {"role": "user", "content": f"""
  Act like the best teacher in the world. Check the information well before you give it to them. Find the answer to the question given below and explain the solution in a few sentences. Also, the question is never wrong. It always has an answer: {text} """},
          ]
  )

    result = ''
    for choice in response.choices:
        result += choice.message.content
        print(choice.message.content)
    return result


# Define the OCR function
def ocr_space_file(filename, overlay=False, language='tur'):
    payload = {'isOverlayRequired': overlay,
               'apikey': API_KEY,
               'language': language
               }
    with open(filename, 'rb') as f:
        r = requests.post('https://api.ocr.space/parse/image',
                          files={filename: f},
                          data=payload)
    return r.json()

# Define the function to handle image messages
def handle_image(update, context):
    # Get the file ID and download the image
    file_id = update.message.photo[-1].file_id
    file = context.bot.get_file(file_id)
    file_path = os.path.join('downloads', f'{file_id}.jpg')
    file.download(file_path)

    # Perform OCR on the downloaded image
    result = ocr_space_file(file_path)
    # Extract the parsed text from the OCR result
    parsed_text = result['ParsedResults'][0]['ParsedText']
    parsed_text = create(parsed_text)
    # Reply to the user with the extracted text
    update.message.reply_text(parsed_text)

# Define the function to handle the /start command
def start_command(update, context):
    update.message.reply_text('Welcome! Send us a picture of the problem you want solved. Only text questions are accepted - yet!')

# Set up the Telegram bot

updater = Updater(TOKEN, use_context=True)
dispatcher = updater.dispatcher

# Register the image handler function
image_handler = MessageHandler(Filters.photo, handle_image)
dispatcher.add_handler(image_handler)
# Register the start command handler function

start_handler = CommandHandler('start', start_command)
dispatcher.add_handler(start_handler)
# Start the bot
updater.start_polling()
