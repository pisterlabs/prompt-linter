import telebot
import openai 
from pydub import AudioSegment
from envs import openaikey, proxy_url, telekey, password, user, database

import mysql.connector
import sys

# create log table in database
# CREATE TABLE log (
#     id INT AUTO_INCREMENT PRIMARY KEY,
#     side VARCHAR(255),
#     text TEXT,
#     time TIMESTAMP,
#     other VARCHAR(100)
# );

try:
    conn = mysql.connector.connect(
        host="localhost",
        user=user,
        password=password,
        database=database)
    print("Connected to MySQL server successfully!")
except mysql.connector.Error as error:
    print("Failed to connect to MySQL server: {}".format(error))
    sys.exit(1)

cursor = conn.cursor()

openai.api_key = openaikey 
bot = telebot.TeleBot(telekey)

@bot.message_handler(commands=['start'])
def message_handler_start_main(message):
	msg = bot.send_message(message.chat.id, 'hey hi')
	return

@bot.message_handler(func=lambda message: True)
def echo_all(message):
    if 707939820 == message.chat.id: 
        insert_query = "INSERT INTO log (side, text, time, other) VALUES (%s, %s, NOW(), %s)"
        values = ("user", message.text, "other")
        cursor.execute(insert_query, values)
        conn.commit()
        cursor.execute("SELECT * FROM log WHERE time >= NOW() - INTERVAL 5 MINUTE ORDER BY time;")
        rows = cursor.fetchall()
        delete_query = "DELETE FROM log WHERE time < NOW() - INTERVAL 5 MINUTE;"
        cursor.execute(delete_query)
        conn.commit()
        for row in rows:
            print(row)
        response = openai.ChatCompletion.create(
	        model="gpt-3.5-turbo-0301",
	        messages=[
	                # {"role": "system", "content": "You are a chatbot"},
	                {"role": "system", "content": "You are a Asuna anime girl from SAO anime that talking cute and know russian"},
	                # {"role": "user", "content": message.text},
	                *[{"role": row[1], "content": row[2]} for row in rows],
	            ]
	    )
        result = ''
        for choice in response.choices:
	        result += choice.message.content
        insert_query = "INSERT INTO log (side, text, time, other) VALUES (%s, %s, NOW(), %s)"
        values = ("system", result, "other")
        cursor.execute(insert_query, values)
        conn.commit()
        bot.send_message(message.chat.id,result)
    else:
        bot.send_message(message.chat.id,"This is personal bot for Sepezho. Please, don't use it. Thanks. But you can run your own bot by forking this repo: github.com/sepezho/chatgpt-voice-tg-bot")

@bot.message_handler(content_types=['voice', 'audio'])
def get_audio_messages(message):
	if 707939820 == message.chat.id: 
		file_info = bot.get_file(message.voice.file_id)
		print(file_info.file_path)
		downloaded_file = bot.download_file(file_info.file_path)
		print(downloaded_file)
		with open('user_voice.ogg', 'wb') as new_file:
			new_file.write(downloaded_file)
		AudioSegment.from_file("/Users/sepezho/Work/ai/user_voice.ogg", format="ogg").export("/Users/sepezho/Work/ai/audio.mp3", format="mp3")
		audio_file= open("audio.mp3", "rb")
		transcript = openai.Audio.transcribe("whisper-1",audio_file)
		insert_query = "INSERT INTO log (side, text, time, other) VALUES (%s, %s, NOW(), %s)"
		values = ("user", transcript.text, "other")
		cursor.execute(insert_query, values)
		conn.commit()
		cursor.execute("SELECT * FROM log WHERE time >= NOW() - INTERVAL 5 MINUTE ORDER BY time;")
		rows = cursor.fetchall()
		delete_query = "DELETE FROM log WHERE time < NOW() - INTERVAL 5 MINUTE;"
		cursor.execute(delete_query)
		conn.commit()
		print(transcript.text)
		response = openai.ChatCompletion.create(
		    model="gpt-3.5-turbo-0301",
		    messages=[
		            # {"role": "system", "content": "You are a chatbot"},
	                {"role": "system", "content": "You are a Asuna anime girl from SAO anime that talking cute and know russian"},
		            {"role": "user", "content": transcript.text},
		        ]
		)
		result = ''
		for choice in response.choices:
		    result += choice.message.content
		insert_query = "INSERT INTO log (side, text, time, other) VALUES (%s, %s, NOW(), %s)"
		values = ("system", result, "other")
		cursor.execute(insert_query, values)
		conn.commit()
		bot.send_message(message.chat.id,result)
	else:
		bot.send_message(message.chat.id,"This is personal bot for Sepezho. Please, don't use it. Thanks. But you can run your own bot by forking this repo: github.com/sepezho/chatgpt-voice-tg-bot")

bot.polling()
if cursor:
    cursor.close()
if conn:
    conn.close()
    print("MySQL connection closed.")
