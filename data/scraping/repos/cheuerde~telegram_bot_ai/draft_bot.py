import requests
import os
import sqlite3
import openai
import json
import time
import telegram
from base64 import b64decode

# get the api token from the env varialbe teelgram_api_key
telegram_api_key = os.environ.get("TELEGRAM_API_KEY")
bot = telegram.Bot(telegram_api_key)

openai_api_key = os.environ.get("OPENAI_API_KEY")

sqlite_db_name = "telegram_bot.db"

openai.api_key = openai_api_key

# openai.Model.retrieve("text-davinci-003")

# Set the keywords you want the bot to respond to
keywords = ["hello", "hi", "greetings"]

# Connect to the database
conn = sqlite3.connect(sqlite_db_name)
cursor = conn.cursor()

# Create the messages table if it doesn't exist
cursor.execute("CREATE TABLE IF NOT EXISTS messages (id INTEGER PRIMARY KEY)")

# get the highest message id
cursor.execute("SELECT MAX(id) FROM messages")
max_id = cursor.fetchone()[0]
if max_id is None:
    max_id = 0

# Set the initial value of the offset
offset = max_id - 5

offset = -1

# Continuously check for new messages
while True:

    time.sleep(2)
    # Use the getUpdates method to get the latest updates
    url = f"https://api.telegram.org/bot{telegram_api_key}/getUpdates"
    params = {"offset": offset, "limit": 10}
    response = requests.get(url, params = params, verify = False)
    updates = response.json()["result"]

    # Process each update
    for update in updates:

        message = update.get("message", "")

        if message != "":

          # Get the message ID and update the offset
          message_id = update["message"]["message_id"]
          offset = message_id + 1

          # Check if the message has been processed before
          cursor.execute("SELECT * FROM messages WHERE id=?", (message_id,))
          if cursor.fetchone():
              # If the message has been processed before, skip it
              continue

          chat_id = message["chat"]["id"]
          text = message["text"]

          # no check if the text starts with "/image"
          if text.startswith("/image"):

            try: 

              out = openai.Image.create(
                prompt=text,
                n=1,
                #size="256x256",
                #size="512x512",
                size="1024x1024",
                #response_format="b64_json"
                response_format="url"
              )

              json_object = json.loads(str(out))
              response = json_object['data'][0]['url']

              bot.send_photo(chat_id, response)

            except Exception as e:

                response = "Prompt refused by OpenAI API"

                url = f"https://api.telegram.org/bot{telegram_api_key}/sendMessage"
                data = {
                    "chat_id": chat_id,
                    "text": response
                }
                requests.post(url, data=data)

          else:

            try: 

            # create openai response
              out = openai.Completion.create(
                model="text-davinci-003",
                #model = "text-curie-001",
                prompt=text,
                max_tokens=1000,
                temperature=0.7
              )

              json_object = json.loads(str(out))
              response = json_object['choices'][0]['text']

              url = f"https://api.telegram.org/bot{telegram_api_key}/sendMessage"
              data = {
                  "chat_id": chat_id,
                  "text": response
              }
              requests.post(url, data=data)

            except Exception as et:

              response = "OpenAI error"

              url = f"https://api.telegram.org/bot{telegram_api_key}/sendMessage"
              data = {
                  "chat_id": chat_id,
                  "text": response
              }
              requests.post(url, data=data)

          # If the message has not been processed before, add it to the database
          cursor.execute("INSERT INTO messages (id) VALUES (?)", (message_id,))
          conn.commit()

# Close the database connection
conn.close()
