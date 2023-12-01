# Yrigoyen ChatGPT happy news animatronic 
# Roni Bandini @RoniBandini bandini.medium.com
# August 2023, MIT License

from reader import make_reader
from gtts import gTTS
import sys
import openai
import time
import os
import threading
import random
from pinpong.board import Board, Pin
from unihiker import GUI
from unihiker import Audio

audio = Audio() 
gui = GUI() 
Board().begin() 

model_to_use	="text-davinci-003" # most capable
pwm0 		= Pin(Pin.D22, Pin.PWM)
feed_url 	= "https://www.cbsnews.com/latest/rss/main" 
openai.api_key	= ""
prompt		= "Rewrite this news headline with a joyful and optimistic tone:"

def playWav():
	print('Talking')
	audio.play('speech.wav')

def chatGPT(query):
	response = openai.Completion.create(
		model=model_to_use,
		prompt=query,
		temperature=0.9,
		max_tokens=1000
		)
	return str.strip(response['choices'][0]['text']), response['usage']['total_tokens']

reader = make_reader("db.sqlite")

def add_and_update_feed():
	reader.add_feed(feed_url, exist_ok=True)
	reader.update_feeds()

def download_everything():

	entries = reader.get_entries()
	entries = list(entries)[:10]

	myCounter=1
	myY=190

	for entry in entries:

		img = gui.draw_image(x=0, y=20, w=240, h=320, image='background.png')
		gui.draw_text(x = 120,y=myY,text='Roni Bandini 8/2023 Argentina', font_size=8, origin='top' )
		myY=myY+20

		print("")
		print("Original headline: "+entry.title)

		query=prompt+entry.title
		(res, usage) = chatGPT(query)
		print("Joyful headline: "+res)

		gui.draw_text(x = 120,y=myY,text='News headline:', font_size=8, origin='top' )
		myY=myY+10

		# separate into words
		words = str(entry.title).split() 
		# count words
		howManyWords=len(words)

		myLine=""

		# iterate and prepare 40 char lines
		for x in words:
			if len(x)+len(myLine)<40:
				myLine=myLine+" "+x
			else:
				gui.draw_text(x = 120,y=myY,text=myLine, font_size=8, origin='top' )
				myY=myY+10
				myLine=x	

		# print remaining
		gui.draw_text(x = 120,y=myY,text=myLine, font_size=8, origin='top' )		
		myY=myY+20

		tts = gTTS(res, lang='en-US')
		tts.save("speech.wav")
		gui.draw_text(x = 120,y=myY,text="Talking...", font_size=8, origin='top' )
		myY=myY+20		

		thread1 = threading.Thread(target=playWav)

		thread1.start()

		# wav play delay
		time.sleep(4)

		closed=1

		while thread1.is_alive():

			if closed==1:
				myOpen=random.randrange(195, 200, 2)
				pwm0.write_analog(myOpen) 	
				time.sleep(0.1)
				closed=0
			else:
				myClose=random.randrange(185, 190, 2)
				pwm0.write_analog(myClose) 		
				time.sleep(0.12)
				closed=1

		gui.draw_text(x = 120,y=myY,text="Searching headline...", font_size=8, origin='top' )
		myY=myY+20
		time.sleep(10)
		myCounter=myCounter+1
		myY=200
				
if __name__ =="__main__":

	os.system('clear')
	img = gui.draw_image(x=0, y=20, w=240, h=320, image='background.png')
	gui.draw_text(x = 120,y=190,text='Roni Bandini 8/2023 Argentina', font_size=8, origin='top' )
	print("Yrigoyen ChatGPT based joyful news animatronic started")
	print("v1.0 @ronibandini August 2023")
	print("")

	pwm0.write_analog(185) 
	add_and_update_feed()
	feed = reader.get_feed(feed_url)
	download_everything()

