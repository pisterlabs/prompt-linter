# Project jarvis
import os
os.system('python.exe -m pip install --upgrade pip')
try:
	from tkinter import *
	import tkinter.messagebox
	import tkinter
except:
	os.system('pip install tk')
try:
	import datetime
except:
	os.system ('pip install datetime')
try:
	import sys
except:
	os.system('pip install sys')
try:
	import re
	import urllib.request as ytsearch
except:
	os.system('pip install urllib')		
try:
	import threading
except:
	os.system('pip install threading')
try:
	import random as rd
	from random import *
	import random
except:
	os.system('pip install random')
try:
	import matplotlib.pyplot as pt
except:
	os.system('pip install matplotlib')
try:
	import ctypes
except:
	os.system('pip install ctypes')
try:
	import json
except:
	os.system('pip install json')
try:
	import pyaudio
except:
	os.system('pip install pyaudio')
try:
	import base64
except:
	os.system('pip install base64')
try:
	import smtplib
except:
	os.system('pip install smtplib')
try:
	import pyjokes
except:
	os.system('pip install pyjokes')
try:
	import time
	from time import sleep
except:
	os.system('pip install time')
try:
	import webbrowser
except:
	os.system('pip install webbrowser')
try:
	import pywhatkit
except:
	os.system('pip install pywhatkit')
try:
	from bs4 import BeautifulSoup
except:
	os.system('pip install bs4')
try:
	import requests
except:
	os.system('pip install requests')
try:
	import pyautogui
except:
	os.system('pip install pyautogui')
try:
	import pynput
	from pynput.keyboard import Key , Controller
except:
	os.system('pip install pynput')
try:
	import pyttsx3
	import speech_recognition as sr
	from googletrans import Translator
except:
	os.system('pip install googletrans')
	os.system('pip install SpeechRecognition')
	os.system('pip install pyttsx3')
try:
	import speedtest
	from speedtest import *
except:
	os.system('pip install speedtest')
try:
	import wikipedia
	import wikipedia as googleScrap
except:
	os.system('pip install wikipedia')
try:
	from pymsgbox import password
except:
	os.system('pip install pymsgbox')
try:
	import pyqrcode
	from pyqrcode import QRCode
except:
	os.system('pip install pyqrcode')
try:
	import PyPDF2
except:
	os.system('pip install PyPDF2')
try:
	import wolframalpha
except:
	os.system('pip install wolframalpha')
try:
	import keyboard
except:
	os.system('pip install keyboard')
try:
	import numpy as np
except:
	os.system('pip install numpy')
try:
	import mediapipe as mp
except:
	os.system('pip install mediapipe')
try:
	from collections import deque
except:
	os.system('pip install collections')
try:
	import cv2
except:
	os.system('pip install opencv-python')
try:
	import cvzone
	from cvzone.HandTrackingModule import HandDetector
except:
	os.system('pip install cvzone')
try:
	import openai
except:
	os.system('pip install openai')
try:
	from dotenv import load_dotenv
except:
	os.system('pip install python-dotenv')				
from Brain.AIBrain import *

keyboard = Controller ( )

engine = pyttsx3.init ( 'sapi5' )

voices = engine.getProperty ( 'voices' )  # getting details of current voice

engine.setProperty ( 'voice' , voices [ 0 ].id )

engine.setProperty ( 'rate' , 180 )


def speak ( audio ) :  # here audio is var which contain text
	engine.say ( audio )

	engine.runAndWait ( )


def wish ( ) :
	hour = int ( datetime.datetime.now ( ).hour )
	if hour >= 0 and hour < 12 :
		speak ( "Good morning sir, How can I help you" )
		print ( "Good morning sir, How can I help you" )
	elif hour >= 12 and hour < 15 :
		speak ( "Good afternoon sir, How can I help you" )
		print ( "Good night sir, How can I help you" )
	elif hour >= 15 and hour < 21 :
		speak ( "Good evening sir, How can I help you" )
		print ( "Good evening sir, How can I help you" )
	else :
		speak ( "Good night sir, How can I help you" )
		print ( "Good night sir, How can I help you" )


# now convert audio to text
def takecommand():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        print("Listening.....")
        r.pause_threshold = 1
        r.energy_threshold = 300
        audio = r.listen(source,0,4)

    try:
        print("Understanding..")
        query  = r.recognize_google(audio,language='en-in')
        print(f"You Said: {query}\n")
    except Exception as e:
        print("Say that again")
        return "None"
    return query

def latestnews ( ) :
	api_dict = {
		"business" : "https://newsapi.org/v2/top-headlines?country=in&category=business&apiKey=#here paste your api key" ,
		"entertainment" : "https://newsapi.org/v2/top-headlines?country=in&category=entertainment&apiKey=#here 800ed024379341adaa003b2bea6974e3" ,
		"health" : "https://newsapi.org/v2/top-headlines?country=in&category=health&apiKey=#here 800ed024379341adaa003b2bea6974e3" ,
		"science" : "https://newsapi.org/v2/top-headlines?country=in&category=science&apiKey=#here 800ed024379341adaa003b2bea6974e3" ,
		"sports" : "https://newsapi.org/v2/top-headlines?country=in&category=sports&apiKey=#here 800ed024379341adaa003b2bea6974e3" ,
		"technology" : "https://newsapi.org/v2/top-headlines?country=in&category=technology&apiKey=#here 800ed024379341adaa003b2bea6974e3"
		}

	content = None
	url = None
	speakrrr = 'Which field news do you want, [business] , [health] , [technology], [sports] , [entertainment] , [science]'
	print ( speakrrr )
	field = input("Enter : ")
	for key , value in api_dict.items ( ) :
		if key.lower ( ) in field.lower ( ) :
			url = value
			print ( url )
			print ( "url was found" )
			break
		else :
			url = True
	if url is True :
		print ( "url not found" )

	news = requests.get ( url ).text
	news = json.loads ( news )
	speak ( "Here is the first news." )

	arts = news [ "articles" ]
	for articles in arts :
		article = articles [ "title" ]
		speak ( article )
		print ( article )
		news_url = articles [ "url" ]
		speak ( f"for more info visit: {news_url}" )
		print('Type 1 to continue and 2 to stop')
		a = input("Enter : ")
		if str ( a ) == "1" :
			pass
		elif str ( a ) == "2" :
			break

	print ( "thats all" )
	speak ("that's all")


def focus_graph ( ) :
	file = open ( "focus.txt" , "r" )
	content = file.read ( )
	file.close ( )

	content = content.split ( "," )
	x1 = [ ]
	for i in range ( 0 , len ( content ) ) :
		content [ i ] = float ( content [ i ] )
		x1.append ( i )

	speak ( content )
	print ( content )
	y1 = content

	pt.plot ( x1 , y1 , color = "red" , marker = "o" )
	pt.title ( "YOUR FOCUSED TIME" , fontsize = 16 )
	pt.xlabel ( "Times" , fontsize = 14 )
	pt.ylabel ( "Focus Time" , fontsize = 14 )
	pt.grid ( )
	pt.show ( )


def is_admin ( ) :
	try :
		return ctypes.windll.shell32.IsUserAnAdmin ( )
	except :
		return False


	if is_admin ( ) :
		current_time = datetime.datetime.now ( ).strftime ( "%H:%M" )
		print("Please tell the time you want to end focus mode")
		Stop_time = input("Enter : ")
		a = current_time.replace ( ":" , "." )
		a = float ( a )
		b = Stop_time.replace ( ":" , "." )
		b = float ( b )
		Focus_Time = b - a
		Focus_Time = round ( Focus_Time , 3 )
		host_path = 'C:\Windows\System32\drivers\etc\hosts'
		redirect = '127.0.0.1'

		speak ( current_time )
		print ( current_time )
		time.sleep ( 2 )
		website_list = [ "www.facebook.com" , "facebook.com" , "www.youtube.com" , "youtube.com"]  # Enter the websites that you want to block
		if (current_time < Stop_time) :
			with open ( host_path , "r+" ) as file :  # r+ is writing+ reading
				content = file.read ( )
				time.sleep ( 2 )
				for website in website_list :
					if website in content :
						pass
					else :
						file.write ( f"{redirect} {website}\n" )
						speak ( "DONE" )
						time.sleep ( 1 )
				speak ( "FOCUS MODE TURNED ON !!!!" )
				print ("FOCUS MODE TURNED ON !!!!")

		while True :

			current_time = datetime.datetime.now ( ).strftime ( "%H:%M" )
			website_list = [ "www.facebook.com" , "facebook.com" ]  # Enter the websites that you want to block
			if (current_time >= Stop_time) :
				with open ( host_path , "r+" ) as file :
					content = file.readlines ( )
					file.seek ( 0 )

					for line in content :
						if not any ( website in line for website in website_list ) :
							file.write ( line )

					file.truncate ( )

					speak ( "Websites are unblocked !!" )
					print ( "Websites are UNblocked !!" )
					file = open ( "focus.txt" , "a" )
					file.write ( f",{Focus_Time}" )  # Write a 0 in focus.txt before starting
					file.close ( )
					break

	else :
		ctypes.windll.shell32.ShellExecuteW ( None , "runas" , sys.executable , " ".join ( sys.argv ) , None , 1 )


def sk () :
	try :
		image = cv2.imread ( "C:\\Users\\vedan\\Desktop\\image.png" )
		grey_filter = cv2.cvtColor ( image , cv2.COLOR_BGR2GRAY )
		invert = cv2.bitwise_not ( grey_filter )
		blur = cv2.GaussianBlur ( invert , (21 , 21) , 0 )
		invertedblur = cv2.bitwise_not ( blur )

		sketch_filter = cv2.divide ( grey_filter , invertedblur , scale = 256.0 )
		cv2.imwrite ( "C:\\Users\\vedan\\Desktop\\sketch.png" , sketch_filter )
	except :
		speak ( 'Please make sure that your file is saved at desktop named as image.png.' )
		print ( 'Please make sure that your file is saved at desktop named as image.png.' )


def slay ( ) :
	time = ("Yesterday " , "2 years ago " , "Today " , "Long time ago ")
	person_name = ("Vedant " , "Jaimin " , "Lofter " , "Jack " , "Joker " , "Parikshit " , "Nishu ")
	place = (
	"playground " , "police station " , "Vedant's house " , "forest " , "zoo " , "old garden " , "high school ")
	find = ("trasure " , "ghosts " , "a secret base " , "guns " , "hightech technology ")

	say = rd.choice ( time ) + ", " + rd.choice ( person_name ) + "went to " + rd.choice (
		place ) + "and found " + rd.choice ( find ) + "there. " + "And then he was very astonished."

	print ( say )

def kun ( ) :
	# Giving different arrays to handle colour points of different colour
	bpoints = [ deque ( maxlen = 1024 ) ]
	gpoints = [ deque ( maxlen = 1024 ) ]
	rpoints = [ deque ( maxlen = 1024 ) ]
	ypoints = [ deque ( maxlen = 1024 ) ]

	# These indexes will be used to mark the points in particular arrays of specific colour
	blue_index = 0
	green_index = 0
	red_index = 0
	yellow_index = 0

	# The kernel to be used for dilation purpose
	kernel = np.ones ( (5 , 5) , np.uint8 )

	colors = [ (255 , 0 , 0) , (0 , 255 , 0) , (0 , 0 , 255) , (0 , 255 , 255) ]
	colorIndex = 0

	# Here is code for Canvas setup
	paintWindow = np.zeros ( (471 , 636 , 3) ) + 255
	paintWindow = cv2.rectangle ( paintWindow , (40 , 1) , (140 , 65) , (0 , 0 , 0) , 2 )
	paintWindow = cv2.rectangle ( paintWindow , (160 , 1) , (255 , 65) , (255 , 0 , 0) , 2 )
	paintWindow = cv2.rectangle ( paintWindow , (275 , 1) , (370 , 65) , (0 , 255 , 0) , 2 )
	paintWindow = cv2.rectangle ( paintWindow , (390 , 1) , (485 , 65) , (0 , 0 , 255) , 2 )
	paintWindow = cv2.rectangle ( paintWindow , (505 , 1) , (600 , 65) , (0 , 255 , 255) , 2 )

	cv2.putText ( paintWindow , "CLEAR" , (49 , 33) , cv2.FONT_HERSHEY_SIMPLEX , 0.5 , (0 , 0 , 0) , 2 , cv2.LINE_AA )
	cv2.putText ( paintWindow , "BLUE" , (185 , 33) , cv2.FONT_HERSHEY_SIMPLEX , 0.5 , (0 , 0 , 0) , 2 , cv2.LINE_AA )
	cv2.putText ( paintWindow , "GREEN" , (298 , 33) , cv2.FONT_HERSHEY_SIMPLEX , 0.5 , (0 , 0 , 0) , 2 , cv2.LINE_AA )
	cv2.putText ( paintWindow , "RED" , (420 , 33) , cv2.FONT_HERSHEY_SIMPLEX , 0.5 , (0 , 0 , 0) , 2 , cv2.LINE_AA )
	cv2.putText ( paintWindow , "YELLOW" , (520 , 33) , cv2.FONT_HERSHEY_SIMPLEX , 0.5 , (0 , 0 , 0) , 2 , cv2.LINE_AA )
	cv2.namedWindow ( 'Paint' , cv2.WINDOW_AUTOSIZE )

	# initialize mediapipe
	mpHands = mp.solutions.hands
	hands = mpHands.Hands ( max_num_hands = 1 , min_detection_confidence = 0.7 )
	mpDraw = mp.solutions.drawing_utils

	# Initialize the webcam
	cap = cv2.VideoCapture ( 0 )
	ret = True
	while ret :
		# Read each frame from the webcam
		ret , frame = cap.read ( )

		x , y , c = frame.shape

		# Flip the frame vertically
		frame = cv2.flip ( frame , 1 )
		# hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
		framergb = cv2.cvtColor ( frame , cv2.COLOR_BGR2RGB )

		frame = cv2.rectangle ( frame , (40 , 1) , (140 , 65) , (0 , 0 , 0) , 2 )
		frame = cv2.rectangle ( frame , (160 , 1) , (255 , 65) , (255 , 0 , 0) , 2 )
		frame = cv2.rectangle ( frame , (275 , 1) , (370 , 65) , (0 , 255 , 0) , 2 )
		frame = cv2.rectangle ( frame , (390 , 1) , (485 , 65) , (0 , 0 , 255) , 2 )
		frame = cv2.rectangle ( frame , (505 , 1) , (600 , 65) , (0 , 255 , 255) , 2 )
		cv2.putText ( frame , "CLEAR" , (49 , 33) , cv2.FONT_HERSHEY_SIMPLEX , 0.5 , (0 , 0 , 0) , 2 , cv2.LINE_AA )
		cv2.putText ( frame , "BLUE" , (185 , 33) , cv2.FONT_HERSHEY_SIMPLEX , 0.5 , (0 , 0 , 0) , 2 , cv2.LINE_AA )
		cv2.putText ( frame , "GREEN" , (298 , 33) , cv2.FONT_HERSHEY_SIMPLEX , 0.5 , (0 , 0 , 0) , 2 , cv2.LINE_AA )
		cv2.putText ( frame , "RED" , (420 , 33) , cv2.FONT_HERSHEY_SIMPLEX , 0.5 , (0 , 0 , 0) , 2 , cv2.LINE_AA )
		cv2.putText ( frame , "YELLOW" , (520 , 33) , cv2.FONT_HERSHEY_SIMPLEX , 0.5 , (0 , 0 , 0) , 2 , cv2.LINE_AA )
		# frame = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

		# Get hand landmark prediction
		result = hands.process ( framergb )

		# post process the result
		if result.multi_hand_landmarks :
			landmarks = [ ]
			for handslms in result.multi_hand_landmarks :
				for lm in handslms.landmark :
					# # speak(id, lm)
					# speak(lm.x)
					# speak(lm.y)
					lmx = int ( lm.x * 640 )
					lmy = int ( lm.y * 480 )

					landmarks.append ( [ lmx , lmy ] )

				# Drawing landmarks on frames
				mpDraw.draw_landmarks ( frame , handslms , mpHands.HAND_CONNECTIONS )
			fore_finger = (landmarks [ 8 ] [ 0 ] , landmarks [ 8 ] [ 1 ])
			center = fore_finger
			thumb = (landmarks [ 4 ] [ 0 ] , landmarks [ 4 ] [ 1 ])
			cv2.circle ( frame , center , 3 , (0 , 255 , 0) , -1 )
			if (thumb [ 1 ] - center [ 1 ] < 30) :
				bpoints.append ( deque ( maxlen = 512 ) )
				blue_index += 1
				gpoints.append ( deque ( maxlen = 512 ) )
				green_index += 1
				rpoints.append ( deque ( maxlen = 512 ) )
				red_index += 1
				ypoints.append ( deque ( maxlen = 512 ) )
				yellow_index += 1

			elif center [ 1 ] <= 65 :
				if 40 <= center [ 0 ] <= 140 :  # Clear Button
					bpoints = [ deque ( maxlen = 512 ) ]
					gpoints = [ deque ( maxlen = 512 ) ]
					rpoints = [ deque ( maxlen = 512 ) ]
					ypoints = [ deque ( maxlen = 512 ) ]

					blue_index = 0
					green_index = 0
					red_index = 0
					yellow_index = 0

					paintWindow [ 67 : , : , : ] = 255
				elif 160 <= center [ 0 ] <= 255 :
					colorIndex = 0  # Blue
				elif 275 <= center [ 0 ] <= 370 :
					colorIndex = 1  # Green
				elif 390 <= center [ 0 ] <= 485 :
					colorIndex = 2  # Red
				elif 505 <= center [ 0 ] <= 600 :
					colorIndex = 3  # Yellow
			else :
				if colorIndex == 0 :
					bpoints [ blue_index ].appendleft ( center )
				elif colorIndex == 1 :
					gpoints [ green_index ].appendleft ( center )
				elif colorIndex == 2 :
					rpoints [ red_index ].appendleft ( center )
				elif colorIndex == 3 :
					ypoints [ yellow_index ].appendleft ( center )
		# Append the next deques when nothing is detected to avois messing up
		else :
			bpoints.append ( deque ( maxlen = 512 ) )
			blue_index += 1
			gpoints.append ( deque ( maxlen = 512 ) )
			green_index += 1
			rpoints.append ( deque ( maxlen = 512 ) )
			red_index += 1
			ypoints.append ( deque ( maxlen = 512 ) )
			yellow_index += 1

		# Draw lines of all the colors on the canvas and frame
		points = [ bpoints , gpoints , rpoints , ypoints ]
		# for j in range(len(points[0])):
		#         for k in range(1, len(points[0][j])):
		#             if points[0][j][k - 1] is None or points[0][j][k] is None:
		#                 continue
		#             cv2.line(paintWindow, points[0][j][k - 1], points[0][j][k], colors[0], 2)
		for i in range ( len ( points ) ) :
			for j in range ( len ( points [ i ] ) ) :
				for k in range ( 1 , len ( points [ i ] [ j ] ) ) :
					if points [ i ] [ j ] [ k - 1 ] is None or points [ i ] [ j ] [ k ] is None :
						continue
					cv2.line ( frame , points [ i ] [ j ] [ k - 1 ] , points [ i ] [ j ] [ k ] , colors [ i ] , 2 )
					cv2.line ( paintWindow , points [ i ] [ j ] [ k - 1 ] , points [ i ] [ j ] [ k ] , colors [ i ] ,
							   2 )

		cv2.imshow ( "Output" , frame )
		cv2.imshow ( "Paint" , paintWindow )

		if cv2.waitKey ( 1 ) == ord ( 'q' ) :
			break

	# release the webcam and destroy all active windows
	cap.release ( )
	cv2.destroyAllWindows ( )


def WolfRamAlpha ( query ) :
	apikey = "PPUL39-J3Q8UJ88TH"
	requester = wolframalpha.Client ( apikey )
	requested = requester.query ( query )

	try :
		answer = next ( requested.results ).text
		return answer
	except :
		speak ( "The value is not answerable" )
		print("The Value is not answerable")


def Calc ( query ) :
	Term = str ( query )
	Term = Term.replace ( "jarvis" , "" )
	Term = Term.replace ( "multiply" , "*" )
	Term = Term.replace ( "plus" , "+" )
	Term = Term.replace ( "minus" , "-" )
	Term = Term.replace ( "divide" , "/" )

	Final = str ( Term )
	try :
		result = WolfRamAlpha ( Final )
		print ( f"{result}" )
		print ( result )

	except :
		print ( "The value is not answerable" )


def kkun () :
	print('Enter the url of the qrcode')
	s = input("Enter : ")
	n = 'qr'
	n = n + ".svg"
	url = pyqrcode.create ( s )
	url.svg ( n, scale = 8 )


def volumeup ( ) :
	for i in range ( 5 ) :
		keyboard.press ( Key.media_volume_up )
		keyboard.release ( Key.media_volume_up )
		sleep ( 0.1 )


def volumedown ( ) :
	for i in range ( 5 ) :
		keyboard.press ( Key.media_volume_down )
		keyboard.release ( Key.media_volume_down )
		sleep ( 0.1 )


def tic ( ) :
	def single_player ( ) :
		tk = Tk ( )
		tk.title ( "Single Player" )
		global bclick , flag , player2_name , player1_name , playerb , pa , list1 , fg , flag1 , pm , pe
		pe = StringVar ( )
		playerb = StringVar ( )
		p1 = StringVar ( )
		list1 = [ 1 , 2 , 3 , 4 , 5 , 6 , 7 , 8 , 9 ]
		fg = 0
		flag1 = 0

		player1_name = Entry ( tk , textvariable = p1 , bd = 5 )
		player1_name.grid ( row = 1 , column = 1 , columnspan = 8 )
		# playerb = player2_name.get() + " Wins!"
		pe = player1_name.get ( ) + " Wins!"

		# speak(pm)

		bclick = True
		flag = 0

		def disableButton ( ) :
			button1.configure ( state = DISABLED )
			button2.configure ( state = DISABLED )
			button3.configure ( state = DISABLED )
			button4.configure ( state = DISABLED )
			button5.configure ( state = DISABLED )
			button6.configure ( state = DISABLED )
			button7.configure ( state = DISABLED )
			button8.configure ( state = DISABLED )
			button9.configure ( state = DISABLED )

		def search ( r ) :
			for i in list1 :
				if i == r :
					return True
			else :
				return False

		def ran ( ) :
			return random.choice ( list1 )

		def btn1 ( self , x ) :
			global flag1
			if search ( x ) :
				print ( x )
				self [ "text" ] = "X"
				list1.remove ( x )
				fg = 1
				flag1 = flag1 + 1
			else :
				self [ "text" ] = "O"
				flag1 = flag1 + 1

			checkForWin ( )
			if fg == 1 :
				kw ( ran ( ) )

		def btn2 ( self , x ) :
			global flag1
			if search ( x ) :
				self [ "text" ] = "X"
				list1.remove ( x )
				fg = 1
				flag1 = flag1 + 1
			else :
				self [ "text" ] = "O"
				flag1 = flag1 + 1

			checkForWin ( )
			if fg == 1 :
				kw ( ran ( ) )

		def btn3 ( self , x ) :
			global flag1
			if search ( x ) :
				self [ "text" ] = "X"
				list1.remove ( x )
				fg = 1
				flag1 = flag1 + 1
			else :
				self [ "text" ] = "O"
				flag1 = flag1 + 1

			checkForWin ( )
			if fg == 1 :
				kw ( ran ( ) )

		def btn4 ( self , x ) :
			global flag1
			if search ( x ) :
				self [ "text" ] = "X"
				list1.remove ( x )
				fg = 1
				flag1 = flag1 + 1
			else :
				self [ "text" ] = "O"
				flag1 = flag1 + 1

			checkForWin ( )
			if fg == 1 :
				kw ( ran ( ) )

		def btn5 ( self , x ) :
			global flag1
			if search ( x ) :
				self [ "text" ] = "X"
				list1.remove ( x )
				fg = 1
				flag1 = flag1 + 1
			else :
				self [ "text" ] = "O"
				flag1 = flag1 + 1

			checkForWin ( )
			if fg == 1 :
				kw ( ran ( ) )

		def btn6 ( self , x ) :
			global flag1
			if search ( x ) :
				self [ "text" ] = "X"
				list1.remove ( x )
				fg = 1
				flag1 = flag1 + 1
			else :
				self [ "text" ] = "O"
				flag1 = flag1 + 1
			checkForWin ( )
			if fg == 1 :
				kw ( ran ( ) )

		def btn7 ( self , x ) :
			global flag1
			if search ( x ) :
				self [ "text" ] = "X"
				list1.remove ( x )
				fg = 1
				flag1 = flag1 + 1
			else :
				self [ "text" ] = "O"
				flag1 = flag1 + 1
			checkForWin ( )
			if fg == 1 :
				kw ( ran ( ) )

		def btn8 ( self , x ) :
			global flag1
			if search ( x ) :
				self [ "text" ] = "X"
				list1.remove ( x )
				fg = 1
				flag1 = flag1 + 1
			else :
				self [ "text" ] = "O"
				flag1 = flag1 + 1
			checkForWin ( )
			if fg == 1 :
				kw ( ran ( ) )

		def btn9 ( self , x ) :
			global flag1
			if search ( x ) :
				self [ "text" ] = "X"
				list1.remove ( x )
				fg = 1
				flag1 = flag1 + 1
			else :
				self [ "text" ] = "O"
				flag1 = flag1 + 1

			checkForWin ( )
			if fg == 1 :
				kw ( ran ( ) )

		def kw ( r ) :
			print ( flag1 )
			if r == 1 :
				list1.remove ( r )
				fg = 0
				button1.invoke ( )
			elif r == 2 :
				list1.remove ( r )
				fg = 0
				button2.invoke ( )
			elif r == 3 :
				list1.remove ( r )
				fg = 0
				button3.invoke ( )
			elif r == 4 :
				list1.remove ( r )
				fg = 0
				button4.invoke ( )
			elif r == 5 :
				list1.remove ( r )
				fg = 0
				button5.invoke ( )
			elif r == 6 :
				list1.remove ( r )
				fg = 0
				button6.invoke ( )
			elif r == 7 :
				list1.remove ( r )
				fg = 0
				button7.invoke ( )
			elif r == 8 :
				list1.remove ( r )
				fg = 0
				button8.invoke ( )
			elif r == 9 :
				list1.remove ( r )
				fg = 0
				button9.invoke ( )

			checkForWin ( )

		def checkForWin ( ) :
			global pe
			if (button1 [ 'text' ] == 'X' and button2 [ 'text' ] == 'X' and
					button3 [ 'text' ] == 'X' or
					button4 [ 'text' ] == 'X' and button5 [ 'text' ] == 'X' and
					button6 [ 'text' ] == 'X' or
					button7 [ 'text' ] == 'X' and button8 [ 'text' ] == 'X' and
					button9 [ 'text' ] == 'X' or
					button1 [ 'text' ] == 'X' and button5 [ 'text' ] == 'X' and
					button9 [ 'text' ] == 'X' or
					button3 [ 'text' ] == 'X' and button5 [ 'text' ] == 'X' and
					button7 [ 'text' ] == 'X' or
					button1 [ 'text' ] == 'X' and button4 [ 'text' ] == 'X' and
					button7 [ 'text' ] == 'X' or
					button1 [ 'text' ] == 'X' and button4 [ 'text' ] == 'X' and
					button7 [ 'text' ] == 'X' or
					button2 [ 'text' ] == 'X' and button5 [ 'text' ] == 'X' and
					button8 [ 'text' ] == 'X' or
					button3 [ 'text' ] == 'X' and button6 [ 'text' ] == 'X' and
					button9 [ 'text' ] == 'X') :
				disableButton ( )

				tkinter.messagebox.showinfo ( "Tic-Tac-Toe" , "YOU WON!" )
				tk.destroy ( )



			elif (flag1 == 9) :
				tkinter.messagebox.showinfo ( "Tic-Tac-Toe" , "It is a Tie" )
				tk.destroy ( )

			elif (button1 [ 'text' ] == 'O' and button2 [ 'text' ] == 'O' and
				  button3 [ 'text' ] == 'O' or
				  button4 [ 'text' ] == 'O' and button5 [ 'text' ] == 'O' and
				  button6 [ 'text' ] == 'O' or
				  button7 [ 'text' ] == 'O' and button8 [ 'text' ] == 'O' and
				  button9 [ 'text' ] == 'O' or
				  button1 [ 'text' ] == 'O' and button5 [ 'text' ] == 'O' and
				  button9 [ 'text' ] == 'O' or
				  button3 [ 'text' ] == 'O' and button5 [ 'text' ] == 'O' and
				  button7 [ 'text' ] == 'O' or
				  button1 [ 'text' ] == 'O' and button4 [ 'text' ] == 'O' and
				  button7 [ 'text' ] == 'O' or
				  button2 [ 'text' ] == 'O' and button5 [ 'text' ] == 'O' and
				  button8 [ 'text' ] == 'O' or
				  button3 [ 'text' ] == 'O' and button6 [ 'text' ] == 'O' and
				  button9 [ 'text' ] == 'O' or
				  button7 [ 'text' ] == 'O' and button6 [ 'text' ] == 'O' and
				  button9 [ 'text' ] == 'O') :
				disableButton ( )
				tkinter.messagebox.showinfo ( "Tic-Tac-Toe" , 'COMPUTER WINS!' )
				tk.destroy ( )

		label = Label ( tk , text = "Player 1:" , font = 'Times 20 bold' , bg = 'white' ,
						fg = 'black' , height = 1 , width = 8 )
		label.grid ( row = 1 , column = 0 )

		button1 = Button ( tk , text = " " , font = 'Times 20 bold' , bg = 'gray' ,
						   fg = 'white' , height = 4 , width = 8 , command = lambda : btn1 ( button1 , 1 ) )
		button1.grid ( row = 3 , column = 0 )

		button2 = Button ( tk , text = ' ' , font = 'Times 20 bold' , bg = 'gray' ,
						   fg = 'white' , height = 4 , width = 8 , command = lambda : btn2 ( button2 , 2 ) )
		button2.grid ( row = 3 , column = 1 )

		button3 = Button ( tk , text = ' ' , font = 'Times 20 bold' , bg = 'gray' ,
						   fg = 'white' , height = 4 , width = 8 , command = lambda : btn3 ( button3 , 3 ) )
		button3.grid ( row = 3 , column = 2 )

		button4 = Button ( tk , text = ' ' , font = 'Times 20 bold' , bg = 'gray' ,
						   fg = 'white' , height = 4 , width = 8 , command = lambda : btn4 ( button4 , 4 ) )
		button4.grid ( row = 4 , column = 0 )

		button5 = Button ( tk , text = ' ' , font = 'Times 20 bold' , bg = 'gray' ,
						   fg = 'white' , height = 4 , width = 8 , command = lambda : btn5 ( button5 , 5 ) )
		button5.grid ( row = 4 , column = 1 )

		button6 = Button ( tk , text = ' ' , font = 'Times 20 bold' , bg = 'gray' ,
						   fg = 'white' , height = 4 , width = 8 , command = lambda : btn6 ( button6 , 6 ) )
		button6.grid ( row = 4 , column = 2 )

		button7 = Button ( tk , text = ' ' , font = 'Times 20 bold' , bg = 'gray' ,
						   fg = 'white' , height = 4 , width = 8 , command = lambda : btn7 ( button7 , 7 ) )
		button7.grid ( row = 5 , column = 0 )

		button8 = Button ( tk , text = ' ' , font = 'Times 20 bold' , bg = 'gray' ,
						   fg = 'white' , height = 4 , width = 8 , command = lambda : btn8 ( button8 , 8 ) )
		button8.grid ( row = 5 , column = 1 )

		button9 = Button ( tk , text = ' ' , font = 'Times 20 bold' , bg = 'gray' ,
						   fg = 'white' , height = 4 , width = 8 , command = lambda : btn9 ( button9 , 9 ) )
		button9.grid ( row = 5 , column = 2 )

		tk.mainloop ( )

	def multi_player ( ) :
		global tk
		tk = Tk ( )
		tk.title ( "Tic Tac Toe multiplayer" )
		global bclick , flag , player2_name , player1_name , playerb , pa
		pa = StringVar ( )
		playerb = StringVar ( )
		p1 = StringVar ( )
		p2 = StringVar ( )

		player1_name = Entry ( tk , textvariable = p1 , bd = 5 )
		player1_name.grid ( row = 1 , column = 1 , columnspan = 8 )
		player2_name = Entry ( tk , textvariable = p2 , bd = 5 )
		player2_name.grid ( row = 2 , column = 1 , columnspan = 8 )

		bclick = True
		flag = 0

		def disableButton ( ) :
			button1.configure ( state = DISABLED )
			button2.configure ( state = DISABLED )
			button3.configure ( state = DISABLED )
			button4.configure ( state = DISABLED )
			button5.configure ( state = DISABLED )
			button6.configure ( state = DISABLED )
			button7.configure ( state = DISABLED )
			button8.configure ( state = DISABLED )
			button9.configure ( state = DISABLED )

		def normalButton ( ) :
			button1 [ "state" ] = NORMAL
			button1 [ "state" ] = NORMAL
			button1 [ "state" ] = NORMAL
			button1 [ "state" ] = NORMAL
			button1 [ "state" ] = NORMAL
			button1 [ "state" ] = NORMAL
			button1 [ "state" ] = NORMAL
			button1 [ "state" ] = NORMAL
			button1 [ "state" ] = NORMAL

		def TF ( ) :
			if len ( player1_name.get ( ) ) == 0 and len ( player2_name.get ( ) ) == 0 :
				return False
			else :
				return True

		def btnClick ( buttons ) :

			global bclick , flag , player2_name , player1_name , playerb , pa
			# speak(p1.get())
			# speak(p2.get())
			if (TF ( )) :
				if buttons [ "text" ] == " " and bclick == True :
					buttons [ "text" ] = "X"
					bclick = False
					playerb = player2_name.get ( ) + " Wins!"
					pa = player1_name.get ( ) + " Wins!"
					checkForWin ( )
					flag += 1


				elif buttons [ "text" ] == " " and bclick == False :
					buttons [ "text" ] = "O"
					bclick = True
					checkForWin ( )
					flag += 1
				else :
					tkinter.messagebox.showinfo ( "Tic-Tac-Toe" , "Button already Clicked!" )
			else :
				# disableButton()
				tk.destroy ( )
				tkinter.messagebox.showinfo ( "Tic-Tac-Toe" , "please fill player name before starting the Game" )
			# normalButton()

		def checkForWin ( ) :
			if (button1 [ 'text' ] == 'X' and button2 [ 'text' ] == 'X' and
					button3 [ 'text' ] == 'X' or
					button4 [ 'text' ] == 'X' and button5 [ 'text' ] == 'X' and
					button6 [ 'text' ] == 'X' or
					button7 [ 'text' ] == 'X' and button8 [ 'text' ] == 'X' and
					button9 [ 'text' ] == 'X' or
					button1 [ 'text' ] == 'X' and button5 [ 'text' ] == 'X' and
					button9 [ 'text' ] == 'X' or
					button3 [ 'text' ] == 'X' and button5 [ 'text' ] == 'X' and
					button7 [ 'text' ] == 'X' or
					button1 [ 'text' ] == 'X' and button2 [ 'text' ] == 'X' and
					button3 [ 'text' ] == 'X' or
					button1 [ 'text' ] == 'X' and button4 [ 'text' ] == 'X' and
					button7 [ 'text' ] == 'X' or
					button2 [ 'text' ] == 'X' and button5 [ 'text' ] == 'X' and
					button8 [ 'text' ] == 'X' or
					button7 [ 'text' ] == 'X' and button6 [ 'text' ] == 'X' and
					button9 [ 'text' ] == 'X') :
				disableButton ( )
				tk.destroy ( )
				tkinter.messagebox.showinfo ( "Tic-Tac-Toe" , pa )

			elif (flag == 8) :
				tkinter.messagebox.showinfo ( "Tic-Tac-Toe" , "It is a Tie" )
				tk.destroy ( )

			elif (button1 [ 'text' ] == 'O' and button2 [ 'text' ] == 'O' and
				  button3 [ 'text' ] == 'O' or
				  button4 [ 'text' ] == 'O' and button5 [ 'text' ] == 'O' and
				  button6 [ 'text' ] == 'O' or
				  button7 [ 'text' ] == 'O' and button8 [ 'text' ] == 'O' and
				  button9 [ 'text' ] == 'O' or
				  button1 [ 'text' ] == 'O' and button5 [ 'text' ] == 'O' and
				  button9 [ 'text' ] == 'O' or
				  button3 [ 'text' ] == 'O' and button5 [ 'text' ] == 'O' and
				  button7 [ 'text' ] == 'O' or
				  button1 [ 'text' ] == 'O' and button2 [ 'text' ] == 'O' and
				  button3 [ 'text' ] == 'O' or
				  button1 [ 'text' ] == 'O' and button4 [ 'text' ] == 'O' and
				  button7 [ 'text' ] == 'O' or
				  button2 [ 'text' ] == 'O' and button5 [ 'text' ] == 'O' and
				  button8 [ 'text' ] == 'O' or
				  button7 [ 'text' ] == 'O' and button6 [ 'text' ] == 'O' and
				  button9 [ 'text' ] == 'O') :
				disableButton ( )
				tk.destroy ( )
				tkinter.messagebox.showinfo ( "Tic-Tac-Toe" , playerb )

		buttons = StringVar ( )

		label = Label ( tk , text = "Player 1:" , font = 'Times 20 bold' , bg = 'white' ,
						fg = 'black' , height = 1 , width = 8 )
		label.grid ( row = 1 , column = 0 )

		label = Label ( tk , text = "Player 2:" , font = 'Times 20 bold' , bg = 'white' ,
						fg = 'black' , height = 1 , width = 8 )
		label.grid ( row = 2 , column = 0 )

		button1 = Button ( tk , text = " " , font = 'Times 20 bold' , bg = 'gray' ,
						   fg = 'white' , height = 4 , width = 8 , command = lambda : btnClick ( button1 ) )
		button1.grid ( row = 3 , column = 0 )

		button2 = Button ( tk , text = ' ' , font = 'Times 20 bold' , bg = 'gray' ,
						   fg = 'white' , height = 4 , width = 8 , command = lambda : btnClick ( button2 ) )
		button2.grid ( row = 3 , column = 1 )

		button3 = Button ( tk , text = ' ' , font = 'Times 20 bold' , bg = 'gray' ,
						   fg = 'white' , height = 4 , width = 8 , command = lambda : btnClick ( button3 ) )
		button3.grid ( row = 3 , column = 2 )

		button4 = Button ( tk , text = ' ' , font = 'Times 20 bold' , bg = 'gray' ,
						   fg = 'white' , height = 4 , width = 8 , command = lambda : btnClick ( button4 ) )
		button4.grid ( row = 4 , column = 0 )

		button5 = Button ( tk , text = ' ' , font = 'Times 20 bold' , bg = 'gray' ,
						   fg = 'white' , height = 4 , width = 8 , command = lambda : btnClick ( button5 ) )
		button5.grid ( row = 4 , column = 1 )

		button6 = Button ( tk , text = ' ' , font = 'Times 20 bold' , bg = 'gray' ,
						   fg = 'white' , height = 4 , width = 8 , command = lambda : btnClick ( button6 ) )
		button6.grid ( row = 4 , column = 2 )

		button7 = Button ( tk , text = ' ' , font = 'Times 20 bold' , bg = 'gray' ,
						   fg = 'white' , height = 4 , width = 8 , command = lambda : btnClick ( button7 ) )
		button7.grid ( row = 5 , column = 0 )

		button8 = Button ( tk , text = ' ' , font = 'Times 20 bold' , bg = 'gray' ,
						   fg = 'white' , height = 4 , width = 8 , command = lambda : btnClick ( button8 ) )
		button8.grid ( row = 5 , column = 1 )

		button9 = Button ( tk , text = ' ' , font = 'Times 20 bold' , bg = 'gray' ,
						   fg = 'white' , height = 4 , width = 8 , command = lambda : btnClick ( button9 ) )
		button9.grid ( row = 5 , column = 2 )

		tk.mainloop ( )

	def main ( ) :
		root = Tk ( )

		root.geometry ( '400x500' )
		root.title ( "Tic Tac Toe " )
		button_mp = Button
		f = Frame ( root , height = 70 , width = 1200 , bg = 'crimson' )
		# f.propagate(0)
		f.pack ( )

		h = Frame ( root , height = 70 , width = 1200 )
		# f.propagate(0)
		h.pack ( )
		g = Frame ( root , width = 1200 )
		# f.propagate(0)
		g.pack ( )
		v = Frame ( root , width = 1200 )
		# f.propagate(0)
		v.pack ( )
		q = Label ( v , text = '' , fg = 'red' , height = 3 )
		q.pack ( )

		k = Frame ( root , width = 1200 )
		# f.propagate(0)
		k.pack ( )
		m = Frame ( root , width = 1200 )
		# f.propagate(0)
		m.pack ( )
		i = Label ( m , text = '' , fg = 'red' , height = 3 )
		i.pack ( )

		button2 = Button ( g , bg = 'lightblue' , height = 2 , width = 20 , text = 'Multiplayer' , fg = 'red' , bd = 3 ,
						   command = multi_player )
		# button2.place(bordermode=INSIDE)
		button2.grid ( row = 3 , column = 1 )

		button3 = Button ( k , bg = 'lightblue' , height = 2 , width = 20 , text = 'SinglePlayer' , fg = 'red' ,
						   bd = 3 , command = single_player )
		# button2.place(bordermode=INSIDE)
		button3.grid ( row = 3 , column = 1 )

		root.mainloop ( )

	main ( )


def rock ( ) :
	cap = cv2.VideoCapture ( 0 )
	cap.set ( 3 , 640 )
	cap.set ( 4 , 480 )

	detector = HandDetector ( maxHands = 1 )

	timer = 0
	stateResult = False
	startGame = False
	scores = [ 0 , 0 ]  # [AI, Player]

	while True :
		imgBG = cv2.imread ( "Resources/BG.png" )
		success , img = cap.read ( )

		imgScaled = cv2.resize ( img , (0 , 0) , None , 0.875 , 0.875 )
		imgScaled = imgScaled [ : , 80 :480 ]

		# Find Hands
		hands , img = detector.findHands ( imgScaled )  # with draw

		if startGame :

			if stateResult is False :
				timer = time.time ( ) - initialTime
				cv2.putText ( imgBG , str ( int ( timer ) ) , (605 , 435) , cv2.FONT_HERSHEY_PLAIN , 6 ,
							  (255 , 0 , 255) ,
							  4 )

				if timer > 3 :
					stateResult = True
					timer = 0

					if hands :
						playerMove = None
						hand = hands [ 0 ]
						fingers = detector.fingersUp ( hand )
						if fingers == [ 0 , 0 , 0 , 0 , 0 ] :
							playerMove = 1
						if fingers == [ 1 , 1 , 1 , 1 , 1 ] :
							playerMove = 2
						if fingers == [ 0 , 1 , 1 , 0 , 0 ] :
							playerMove = 3

						randomNumber = random.randint ( 1 , 3 )
						imgAI = cv2.imread ( f'Resources/{randomNumber}.png' , cv2.IMREAD_UNCHANGED )
						imgBG = cvzone.overlayPNG ( imgBG , imgAI , (149 , 310) )

						# Player Wins
						if (playerMove == 1 and randomNumber == 3) or \
								(playerMove == 2 and randomNumber == 1) or \
								(playerMove == 3 and randomNumber == 2) :
							scores [ 1 ] += 1

						# AI Wins
						if (playerMove == 3 and randomNumber == 1) or \
								(playerMove == 1 and randomNumber == 2) or \
								(playerMove == 2 and randomNumber == 3) :
							scores [ 0 ] += 1

		imgBG [ 234 :654 , 795 :1195 ] = imgScaled

		if stateResult :
			imgBG = cvzone.overlayPNG ( imgBG , imgAI , (149 , 310) )

		cv2.putText ( imgBG , str ( scores [ 0 ] ) , (410 , 215) , cv2.FONT_HERSHEY_PLAIN , 4 , (255 , 255 , 255) , 6 )
		cv2.putText ( imgBG , str ( scores [ 1 ] ) , (1112 , 215) , cv2.FONT_HERSHEY_PLAIN , 4 , (255 , 255 , 255) , 6 )

		# cv2.imshow("Image", img)
		cv2.imshow ( "BG" , imgBG )
		# cv2.imshow("Scaled", imgScaled)

		key = cv2.waitKey ( 1 )
		if key == ord ( 's' ) :
			startGame = True
			initialTime = time.time ( )
			stateResult = False
		if key == ord ( 'e' ) :
			exit ( )


def tkk ( ) :
	key = cv2.waitKey ( 1 )
	root = Tk ( )
	root.geometry ( f"2000x750" )
	root.configure ( bg = '#071e26' )
	root.title ( 'Select Game' )
	l = Label ( root , text = "Select Game" , font = 'space' )
	l.pack ( side = 'top' )
	btn = tkinter.Button ( root , text = 'Rock Paper Scissors' , command = rock )
	btn.pack ( side = 'top' )
	btn2 = tkinter.Button ( root , text = 'Tic Tac Toe' , command = tic )
	btn2.pack ( side = 'bottom' )
	root.wm_iconbitmap ( 'icog.ico' )
	if key == ord ( 'e' ) :
		exit ( )
	root.mainloop ( )


def sendEmail ( to , content ) :
	server = smtplib.SMTP ( 'smtp.gmail.com' , 587 )
	server.ehlo ( )
	server.starttls ( )
	server.login ( 'vedantgandhi05@gmail.com' , 'hoernkwsszpzaseu' )
	server.sendmail ( 'vedantgandhi05@gmail.com' , to , content )
	server.close ( )


# for main function

def run_module ( self ) :
	root = Tk ( )
	root.geometry ( '500x300' )
	root.resizable ( 0 , 0 )

	# title of the window
	root.title ( "Encode Decode the Text" )

	# label

	Label ( root , text = 'ENCODE DECODE' , font = 'arial 20 bold' ).pack ( )
	Label ( root , text = 'Data Encryption' , font = 'arial 20 bold' ).pack ( side = BOTTOM )

	# define variables

	Text = StringVar ( )
	private_key = StringVar ( )
	mode = StringVar ( )
	Result = StringVar ( )

	#######define function#####

	# function to encode

	def Encode ( key , message ) :
		enc = [ ]
		for i in range ( len ( message ) ) :
			key_c = key [ i % len ( key ) ]
			enc.append ( chr ( (ord ( message [ i ] ) + ord ( key_c )) % 256 ) )

		return base64.urlsafe_b64encode ( "".join ( enc ).encode ( ) ).decode ( )

	# function to decode

	def Decode ( key , message ) :
		dec = [ ]
		message = base64.urlsafe_b64decode ( message ).decode ( )
		for i in range ( len ( message ) ) :
			key_c = key [ i % len ( key ) ]
			dec.append ( chr ( (256 + ord ( message [ i ] ) - ord ( key_c )) % 256 ) )

		return "".join ( dec )

	# function to set mode
	def Mode ( ) :
		if (mode.get ( ) == 'e') :
			Result.set ( Encode ( private_key.get ( ) , Text.get ( ) ) )
		elif (mode.get ( ) == 'd') :
			Result.set ( Decode ( private_key.get ( ) , Text.get ( ) ) )
		else :
			Result.set ( 'Invalid Mode' )

	# Function to exit window

	def Exit ( ) :
		root.destroy ( )

	# Function to reset
	def Reset ( ) :
		Text.set ( "" )
		private_key.set ( "" )
		mode.set ( "" )
		Result.set ( "" )

	#################### Label and Button #############

	# Message
	Label ( root , font = 'arial 12 bold' , text = 'MESSAGE' ).place ( x = 60 , y = 60 )
	Entry ( root , font = 'arial 10' , textvariable = Text , bg = 'ghost white' ).place ( x = 290 , y = 60 )

	# key
	Label ( root , font = 'arial 12 bold' , text = 'KEY' ).place ( x = 60 , y = 90 )
	Entry ( root , font = 'arial 10' , textvariable = private_key , bg = 'ghost white' ).place ( x = 290 , y = 90 )

	# mode
	Label ( root , font = 'arial 12 bold' , text = 'MODE(e-encode, d-decode)' ).place ( x = 60 , y = 120 )
	Entry ( root , font = 'arial 10' , textvariable = mode , bg = 'ghost white' ).place ( x = 290 , y = 120 )

	# result
	Entry ( root , font = 'arial 10 bold' , textvariable = Result , bg = 'ghost white' ).place ( x = 290 , y = 150 )

	######result button
	Button ( root , font = 'arial 10 bold' , text = 'RESULT' , padx = 2 , bg = 'LightGray' , command = Mode ).place (
		x = 60 , y = 150 )

	# reset button
	Button ( root , font = 'arial 10 bold' , text = 'RESET' , width = 6 , command = Reset , bg = 'LimeGreen' ,
			 padx = 2 ).place ( x = 80 , y = 190 )

	# exit button
	Button ( root , font = 'arial 10 bold' , text = 'EXIT' , width = 6 , command = Exit , bg = 'OrangeRed' , padx = 2 ,
			 pady = 2 ).place ( x = 180 , y = 190 )
	root.mainloop ( )


def yun ( ) :
	speak ( 'Do you want recorded audio or live audio? Yes for recorded and no for live audio' )
	q = input("Enter : ")
	if q == 'yes' :
		try :
			with open ( 'C:\\Users\\vedan\\Desktop\\book.pdf' , 'rb' ) as book :

				full_text = ""

				reader = PyPDF2.PdfFileReader ( book )

				for page in range (reader.numPages) :
					next_page = reader.getPage (page)
					content = next_page.extractText ( )
					full_text += content

				engine.save_to_file ( full_text , "C:\\Users\\vedan\\Desktop\\audio_book.mp3" )
				engine.runAndWait ( )

		except :
			engine.say (
				"Please make sure that you have your pdf file named as book.pdf at desktop and then run again." )

	if q == 'no' :
		try :
			with open ( 'C:\\Users\\vedan\\Desktop\\book.pdf' , 'rb' ) as book :

				full_text = ""

				reader = PyPDF2.PdfFileReader ( book )

				for page in range ( reader.numPages ) :
					next_page = reader.getPage (page)
					content = next_page.extractText ( )
					full_text += content

				speak ( full_text )

		except :
			engine.say (
				"Please make sure that you have your pdf file named as book.pdf at desktop and then run again." )


def call ( person ) :
	call_book = {'papa' : '**********' , 'mom' : '**********'}  # ------------- List of phone number
	if person in call_book :  # ------------------------------ Searching the call book
		ph_no = call_book [ person ]  # ------------------------ Phone no. of the person

		command2 = 'adb shell am start -a android.intent.action.CALL -d tel:+91' + ph_no  # ----cmd. to make call
		command3 = 'adb shell input tap *** ****'  # --------------- cmd. to tap the speaker button
		command1 = 'adb connect ***.***.*.*:****'
		speak ( 'calling.. ' + person )
		os.system ( command1 )
		time.sleep ( 2 )
		os.system ( command2 )  # ----------------------- executing the cmd
		time.sleep ( 2 )
		os.system ( command3 )
	else :
		speak ( 'no contact found' )
		print("NO Contacts FOUND")

def task ( ) :
	if __name__ == "__main__" :
		wish ( )

		while True :
			# if 1:
			query = takecommand().lower ( )  # Converting user query into lower case

			if 'open' in query :

				name = query.replace ( "open " , "" )

				NameA = str ( name )

				if 'youtube' in NameA :

					webbrowser.open ( "https://www.youtube.com/" )

				elif 'instagram' in NameA :

					webbrowser.open ( "https://www.instagram.com/" )

				else :

					string = "https://www." + NameA + ".com"

					string_2 = string.replace ( " " , "" )

					webbrowser.open ( string_2 )

			elif 'wikipedia' in query :
				speak ( "Searching from wikipedia...." )
				query = query.replace ( "wikipedia" , "" )
				query = query.replace ( "search wikipedia" , "" )
				query = query.replace ( "jarvis" , "" )
				results = wikipedia.summary ( query , sentences = 2 )
				speak ( "According to wikipedia.." )
				speak ( results )
				print ( results )

			elif "google" in query :
				query = query.replace ( "jarvis" , "" )
				query = query.replace ( "google search" , "" )
				query = query.replace ( "google" , "" )
				speak ( "This is what I found on google" )

				try :
					pywhatkit.search ( query )
					result = googleScrap.summary ( query , 1 )
					speak ( result )
					print ( result )

				except :
					speak ( "No speakable output available" )

			elif 'time' in query :
				strTime = datetime.datetime.now ( ).strftime ( "%H:%M" )
				speak ( f"Sir, the time is {strTime}" )

			elif 'draw' in query :
				kun ( )

			elif 'youtube' in query :
				query = query.replace ( 'youtube' , '' )
				speak ( 'Playing' + query )
				pywhatkit.playonyt ( query )

			elif 'open youtube' in query or "open video online" in query :
				webbrowser.open ( "www.youtube.com" )
				speak ( "opening youtube" )
			elif 'open github' in query :
				webbrowser.open ( "https://www.github.com" )
				speak ( "opening github" )
			elif 'open facebook' in query :
				webbrowser.open ( "https://www.facebook.com" )
				speak ( "opening facebook" )
			elif 'open instagram' in query :
				webbrowser.open ( "https://www.instagram.com" )
				speak ( "opening instagram" )
			elif 'open google' in query :
				webbrowser.open ( "https://www.google.com" )
				speak ( "opening google" )

			elif 'jokes time' in query or 'jokes' in query :
				joke = pyjokes.get_joke ( )
				speak ( joke )
				print ( joke )

			elif 'play audiobook' in query :
				yun ( )

			elif 'temperature' in query :
				search = "temperature in deesa"
				url = f"https://www.google.com/search?q={search}"
				r = requests.get ( url )
				data = BeautifulSoup ( r.text , "html.parser" )
				temp = data.find ( "div" , class_ = "BNeawe" ).text
				speak ( f"current{search} is {temp}" )
				print ( f"current{search} is {temp}" )

			elif "change password" in query:
				speak("What's the new password")
				new_pw = input("Enter the new password\n")
				new_password = open("password.txt","w")
				new_password.write(new_pw)
				new_password.close()
				speak("Done sir")
				speak(f"Your new password is{new_pw}")

			elif 'weather' in query :
				search = "temperature in deesa"
				url = f"https://www.google.com/search?q={search}"
				r = requests.get ( url )
				data = BeautifulSoup ( r.text , "html.parser" )
				temp = data.find ( "div" , class_ = "BNeawe" ).text
				speak ( f"current{search} is {temp}" )
				print ( f"current{search} is {temp}" )

			elif 'play game' in query or 'game mode' in query :
				tkk()

			elif 'clear' in query :
				speak ( "Clearing" )
				os.system ( 'cls' )

			elif 'bye' in query :
				speak ( "bye bye" )
				exit ( )

			elif 'encode text' in query or 'decode text' in query :
				run_module ( )

			elif 'generate qr' in query or 'generate qrcode' in query :
				kkun()

			elif 'open yahoo' in query :
				webbrowser.open ( "https://www.yahoo.com" )
				speak ( "opening yahoo" )

			elif 'shutdown the system' in query :
				speak ( "Are You sure you want to shutdown" )
				shutdown = takecommand()
				if shutdown == "yes" :
					os.system ( "shutdown /s /t 1" )

				elif shutdown == "no" :
					break

			elif 'call' in query :
				speak ( "Whom Do you want to call?" )
				person = input( "Enter name :" )
				call ( person )

			elif 'story' in query :
				slay ( )

			elif 'all commands' in query :
				speak('Here are all commands')
				os.startfile('commands.txt')

			elif 'internet speed' in query :
				wifi = speedtest.Speedtest ( )
				upload_net = wifi.upload ( ) / 1048576
				download_net = wifi.download ( ) / 1048576
				speak ( "Wifi Upload Speed is" , upload_net )
				print ( "Wifi download speed is " , download_net )
				print ( f"Wifi download speed is {download_net}" )
				speak ( f"Wifi Upload speed is {upload_net}" )

			elif 'open gmail' in query :
				webbrowser.open ( "https://mail.google.com" )
				speak ( "opening google mail" )

			elif 'open snapdeal' in query :
				webbrowser.open ( "https://www.snapdeal.com" )
				speak ( "opening snapdeal" )

			elif 'open amazon' in query or 'shop online' in query :
				webbrowser.open ( "https://www.amazon.com" )
				speak ( "opening amazon" )
			elif 'open flipkart' in query :
				webbrowser.open ( "https://www.flipkart.com" )
				speak ( "opening flipkart" )
			elif 'open ebay' in query :
				webbrowser.open ( "https://www.ebay.com" )
				speak ( "opening ebay" )

			elif 'good bye' in query :
				speak ( "good bye" )
				exit ( )

			elif 'monkey mode' in query :
				speak ( "Entering Monkey Mode" )
				print ( "Entering Monkey Mode" )
				while True :
					monkk = takecommand()
					speak ( monkk )
					print ( monkk )

					if monkk == 'exit monkey mode' :
						task ( )

			elif 'news' in query :
				latestnews ( )

			elif "what's up" in query or 'how are you' in query :
				stMsgs = [ 'Just doing my thing!' , 'I am fine!' , 'Nice!' , 'I am nice and full of energy' ,
						   'i am okey ! How are you' ]
				ans_q = random.choice ( stMsgs )
				speak ( ans_q )
				ans_take_from_user_how_are_you = takecommand()
				if 'fine' in ans_take_from_user_how_are_you or 'happy' in ans_take_from_user_how_are_you or 'okey' in ans_take_from_user_how_are_you :
					speak ( 'okey..' )
				elif 'not' in ans_take_from_user_how_are_you or 'sad' in ans_take_from_user_how_are_you or 'upset' in ans_take_from_user_how_are_you :
					speak ( 'oh sorry..' )

			elif 'sketch' in query :
				sk ( )

			elif 'make you' in query or 'created you' in query or 'develop you' in query :
				ans_m = " For your information Vedant Created me ! I give Lot of Thanks to Him "
				speak ( ans_m )
			elif "who are you" in query or "about you" in query or "your details" in query :
				about = "I am Jarvis an A I based computer program but i can help you lot like a your close friend! "
				speak ( about )
			elif "hello" in query or "hello Jarvis" in query :
				hel = "Hello Vedant Sir ! How May i Help you.."
				speak ( hel )
			elif "your name" in query or "sweat name" in query :
				na_me = "Thanks for Asking my name! My self  Jarvis a, Virtual Artificial Intelligence"
				speak ( na_me )
			elif "you feeling" in query :
				speak ( "feeling Very sweet after meeting with you" )
			elif 'exit' in query or 'abort' in query or 'stop' in query or 'quit' in query :
				ex_exit = 'See you again, Goodbye'
				speak ( ex_exit )
				exit ( )

			elif 'screenshot' in query :
				im = pyautogui.screenshot ( )
				im.save ( "screenshot.jpg" )

			elif 'click my photo' in query :
				pyautogui.press ( "super" )
				pyautogui.typewrite ( "camera" )
				pyautogui.press ( "enter" )
				pyautogui.sleep ( 2 )
				speak ( "Smile please" )
				pyautogui.press ( "enter" )

			elif 'volume up' in query :
				volumeup ( )

			elif 'volume down' in query :
				volumedown ( )

			elif 'focus mode' in query :
				speak( "Are you sure that you want to enter focus mode :- [1 for YES / 2 for NO " )
				a = input("Enter 1 for yes and 2 for no :  ")
				if (a == 1) :
					speak ( "Entering the focus mode...." )
					is_admin ( )
					exit ( )

				else :
					pass

			elif 'show my focus' in query :
				focus_graph ( )

			elif 'email varda' in query :
				try :
					speak ( "What should I say?" )
					content = input("Enter : ")
					to = 'vardagandhi@gmail.com'
					sendEmail ( 'vedantgandhi05@gmail.com' , to , content )
					speak ( "Email has been sent!" )
				except Exception as e :
					print ( e )
					speak ( "Sorry Sir. I was not able to send this email" )

			elif 'email papa' in query :
				try :
					speak ( "What should I say?" )
					content = input("Enter : ")
					to = 'drvjgandhi@gmail.com'
					sendEmail ( 'vedantgandhi05@gmail.com' , to , content )
					speak ( "Email has been sent!" )
				except Exception as e :
					speak ( e )
					speak ( "Sorry Sir. I was not able to send this email" )

			elif 'email' in query :
				try :
					speak ( "Please tell me the email adress" )
					quav = input("Enter : ")

					speak ( "What should I say?" )
					content = input("Enter : ")
					to = quav
					sendEmail ( to , content )
					speak ( "Email has been sent!" )
				except Exception as e :
					print ( e )
					speak ( "Sorry Sir. I was not able to send this email" )

			elif 'calculate' in query :
				query = query.replace ( "calculate" , "" )
				query = query.replace ( "jarvis" , "" )
				Calc ( query )

			elif 'i am fine' in query :
				speak ( "that's great, sir" )

			elif 'how are you' in query :
				speak ( "Perfect, sir" )

			elif 'thank you' in query :
				speak ( "you are welcome, sir" )

			elif "schedule my day" in query:
				tasks = [] #Empty list 
				speak("Do you want to clear old tasks (Plz speak YES or NO)")
				query = takecommand().lower()
				if "yes" in query:
					file = open("tasks.txt","w")
					file.write(f"")
					file.close()
					no_tasks = int(input("Enter the no. of tasks :- "))
					i = 0
					for i in range(no_tasks):
						tasks.append(input("Enter the task :- "))
						file = open("tasks.txt","a")
						file.write(f"{i}. {tasks[i]}\n")
						file.close()
				elif "no" in query:
					i = 0
					no_tasks = int(input("Enter the no. of tasks :- "))
					for i in range(no_tasks):
						tasks.append(input("Enter the task :- "))
						file = open("tasks.txt","a")
						file.write(f"{i}. {tasks[i]}\n")
						file.close()

			elif 'chat with text' in query :
				os.system('python AIBrain.py')
				speak('I am now sleeping tell me wake up jarvis to wake me up')
				osrft = takecommand()
				if osrft == 'wake up jarvis' :
					task()

			elif 'sleep jarvis'	in query :
				kkuttt = takecommand()
				if kkuttt == 'wake up now ' :
					task()
			
for i in range(3):
    a = input("Enter Password to open Jarvis :- ")
    pw_file = open("password.txt","r")
    pw = pw_file.read()
    pw_file.close()
    if (a==pw):
        os.system('cls')
        print("WELCOME SIR")
        break
    elif (i==2 and a!=pw):
        exit()

    elif (a!=pw):
        print("Try Again")
