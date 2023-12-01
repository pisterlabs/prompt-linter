#from squizz import squizznet
#import markiplier
import flet as ft
from flet import *
from pygame import mixer
import os
import openai
from dotenv import load_dotenv

load_dotenv()
openai.organization = "org-F5uP5RO6y4zEyQmPuQLmfyzf"
openai.api_key = os.getenv("OPENAI_API_KEY")
openai.Model.list();

page = ft.Page
qTopic = ft.TextField(autofocus=True, border_color="white")
contextField = ft.TextField(autofocus=True, border_color="white")
question = ''
ans = ''
correctAns = ''
    
def main(page: Page):
	page.title = "SquizzNET™"

	openai.Model.retrieve("gpt-3.5-turbo")
	mixer.init()
	mixer.music.load("squizz.wav")
	mixer.music.set_volume(0.8)
	mixer.music.play()

	def route_change(route):
		# CLEAR ALL PAGE
		page.views.clear()
		page.views.append(
				View(
                      "/", [
                            ft.Row([ft.Text(value="Welcome to SquizzNET™", text_align=ft.TextAlign.RIGHT)], alignment=ft.MainAxisAlignment.CENTER),
                            ft.Row([ft.Image(src=f"ezgif.com-crop.gif")], alignment=ft.MainAxisAlignment.CENTER),
    						ft.Row([ft.ElevatedButton("Press to Squizz", on_click=lambda _: page.go(f"/generate"))], alignment=ft.MainAxisAlignment.CENTER)
        					]    
					)
				)
		if page.route == f"/generate":
			
			qTopic = ft.TextField(autofocus=True, border_color="white")
			contextField = ft.TextField(autofocus=True, border_color="white")
			def makeSquizz(e):
				global question
				question = "Make a multiple choice question about " + qTopic.value
				context = contextField.value
				question = question + " and provide the answer. Please provide 4 answer choices labeled with letters and format the answers with a ) and list the answer after 'Answer: '. Do not preface the questions with a number"
				if context != '':
					question = question + "For context: " + context
				page.go(f"/squizz")
			
			page.views.append(
					View(
                    	f"/generate", [ft.Container(width=1,height=300),
				    	ft.Row([ft.Text(value="Provide a specific topic for your Squizz", text_align=ft.TextAlign.RIGHT)],
            			alignment=ft.MainAxisAlignment.CENTER), 
        				ft.Row([qTopic], alignment=ft.MainAxisAlignment.CENTER),
			            ft.Row([ft.Text(value="Provide additional context for your Squizz", text_align=ft.TextAlign.RIGHT)],
		                alignment=ft.MainAxisAlignment.CENTER),
			            ft.Row([contextField], alignment=ft.MainAxisAlignment.CENTER),
			            ft.Row([ft.ElevatedButton("Generate Squizz", on_click=makeSquizz)], alignment=ft.MainAxisAlignment.CENTER),]
					)
			)
                        
		if page.route == f"/squizz":
			if 'Paralyzer' in question:
				mixer.music.load("soundtrack.mp3")
				mixer.music.play()
			userQst = ''
			answerA = ft.OutlinedButton()
			answerB = ft.OutlinedButton()
			answerC = ft.OutlinedButton()
			answerD = ft.OutlinedButton()
			def aButt(e):
				if ans.lower() == 'a':
					page.go(f"/correct")
				else:
					page.go(f"/incorrect")
			def bButt(e):
				if ans.lower() == 'b':
					page.go(f"/correct")
				else:
					page.go(f"/incorrect")
			def cButt(e):
				if ans.lower() == 'c':
					page.go(f"/correct")
				else:
					page.go(f"/incorrect")
			def dButt(e):
				if ans.lower() == 'd':
					page.go(f"/correct")
				else:
					page.go(f"/incorrect")

			if question != '':
				completion = openai.ChatCompletion.create(
  					model="gpt-3.5-turbo",
  					messages=[
    					{"role": "user", "content": question}
  					]
					)
				print(completion.choices[0].message)
				initText = completion.choices[0].message.content
				i = 0
				while initText[i] != '\n':
					userQst = userQst + initText[i]
					i += 1
				while initText[i] != '\n':
					i += 1
				i += 4
				ansAText= ''
				while initText[i] != '\n':
					ansAText = ansAText + initText[i]
					i += 1
				i += 4
				ansBText= ''
				while initText[i] != '\n':
					ansBText = ansBText + initText[i]
					i += 1
				i += 4
				ansCText= ''
				while initText[i] != '\n':
					ansCText = ansCText + initText[i]
					i += 1
				i += 4
				ansDText= ''
				while initText[i] != '\n':
					ansDText = ansDText + initText[i]
					i += 1
				while initText[i] != ':':
					i += 1
				i += 2
				global ans
				ans = initText[i]
				print(ans)
				global correctAns
				if ans.lower() == 'a':
					correctAns = ansAText
				elif ans.lower() == 'b':
					correctAns = ansBText
				elif ans.lower() == 'c':
					correctAns = ansCText
				elif ans.lower() == 'd':
					correctAns = ansDText
				answerA = ft.OutlinedButton(ansAText, on_click=aButt)
				answerB = ft.OutlinedButton(ansBText, on_click=bButt)
				answerC = ft.OutlinedButton(ansCText, on_click=cButt)
				answerD = ft.OutlinedButton(ansDText, on_click=dButt)

			page.views.append(
					View(
                    	f"/squizz", [ft.Container(width=1,height=300),
				  		ft.Row([ft.Text(value=userQst,text_align=ft.TextAlign.CENTER)], alignment=ft.MainAxisAlignment.CENTER),
						ft.Row([answerA], alignment=ft.MainAxisAlignment.CENTER),
						ft.Row([answerB], alignment=ft.MainAxisAlignment.CENTER),
						ft.Row([answerC], alignment=ft.MainAxisAlignment.CENTER),
						ft.Row([answerD], alignment=ft.MainAxisAlignment.CENTER)]
					)
			)

		if page.route == f"/correct":
			page.views.append(
				View(
                   	f"/correct", [ft.Container(width=1,height=300),
			    	ft.Row([ft.Text(value="Correct!", text_align=ft.TextAlign.RIGHT)],
           			alignment=ft.MainAxisAlignment.CENTER), 
        			ft.Row([ft.ElevatedButton("Create Another Squizz", on_click=lambda _: page.go(f"/generate"))], alignment=ft.MainAxisAlignment.CENTER)]
				)
		)

		if page.route == f"/incorrect":
			incorrText = "The correct answer was: " + correctAns
			page.views.append(
				View(
                   	f"/correct", [ft.Container(width=1,height=300),
			    	ft.Row([ft.Text(value="Sorry, you are incorrect.", text_align=ft.TextAlign.RIGHT)],
           			alignment=ft.MainAxisAlignment.CENTER), 
			    	ft.Row([ft.Text(value=incorrText, text_align=ft.TextAlign.RIGHT)],
           			alignment=ft.MainAxisAlignment.CENTER), 
        			ft.Row([ft.ElevatedButton("Create Another Squizz", on_click=lambda _: page.go(f"/generate"))], alignment=ft.MainAxisAlignment.CENTER)]
				)
		)
                
        
	page.update()

	def view_pop(view):
		page.views.pop()
		myview = page.views[-1]
		page.go(myview.route)

	page.on_route_change = route_change
	page.on_view_pop = view_pop
	page.go(page.route)

ft.app(target=main)