from tkinter import *
from tkinter import ttk, Text
from tkinter import messagebox
from PIL import Image, ImageTk
import pygame
import random
import time
import openai
import requests
import json
import webbrowser


   

def create_layered_rectangle(canvas, x1, y1, x2, y2, layers=3, offset=1, **kwargs):
    for i in range(layers):
        canvas.create_rectangle(
            x1 - i * offset, y1 - i * offset, 
            x2 + i * offset, y2 + i * offset, 
            **kwargs
)



# Function to make objects draggable and check for overlap
def on_drag(event, id):
    canvas.coords(id, event.x, event.y)
    if check_overlap(id, rubrics_obj):
        print("Key successfully placed in Rubrics!")
        messagebox.showinfo("Success", "Riddle Solved! This next step is importan professor! click ok and then send a message to the robot!")
        webbrowser.open('https://mediafiles.botpress.cloud/8ad03b22-87fc-4317-9b85-e60b4790f1ad/webchat/bot.html')  # This
       
        

# Function to check if two objects overlap
def check_overlap(obj1, obj2):
    x1, y1, x2, y2 = canvas.bbox(obj1)
    overlap = canvas.find_overlapping(x1, y1, x2, y2)
    return obj2 in overlap
   

def handle_overlap():
    if check_overlap(key_obj, rubrics_obj):
        print("Key successfully placed in Rubrics!")
  


# Initialize the Tkinter window
root = Tk()
root.configure(background='black')
root.title("Proffesor Ahmed the KEY to getting my final assignment is in the Rubrics")
canvas = Canvas(root, width=800, height=600)
canvas.pack()
response_label = Label(root, text="Initial text")

# Initialize Pygame for audio playback
pygame.init()
pygame.mixer.init()
pygame.mixer.music.load("C:/Users/lordb/OneDrive/Desktop/SAW program/SAWGAME/obi-tape.ogg")
pygame.mixer.music.play(-1)  # -1 means the music will loop indefinitely


# Resizing the Key image
image_key = Image.open("C:/Users/lordb/OneDrive/Desktop/SAW program/SAWGAME/key.png")
image_key = image_key.resize((50, 50), Image.Resampling.NEAREST)
key_img_resized = ImageTk.PhotoImage(image_key)

# Resizing the Rubrics image
image_rubrics = Image.open("C:/Users/lordb/OneDrive/Desktop/SAW program/SAWGAME/rubrics.png")
image_rubrics = image_rubrics.resize((75, 75), Image.Resampling.NEAREST)
rubrics_img_resized = ImageTk.PhotoImage(image_rubrics)

# Load and display the SAW image
saw_img = PhotoImage(file="C:/Users/lordb/OneDrive/Desktop/SAW program/SAWGAME/saw_image.png")  
saw_obj = canvas.create_image(400, 300, image=saw_img)

def move_image():
    y = 300  # Initial y-coordinate
    while True:
        y += random.randint(-5, 5)  # Random jiggle motion
        canvas.coords(saw_obj, 400, y)
        root.update()
        time.sleep(0.1)

# Function to start the image movement
def start_moving_image():
    move_image()

create_layered_rectangle(canvas, 150, 275, 650, 325, fill="black")  #<--- hardest part of the whole program

riddle_text = canvas.create_text(400, 300, text="wHaT iS thE KEY tO bEinG a gOoD pRofESsOR?...... AhMeD?", font=("Arial", 24), fill="white", width=600, justify="center")

# Load and display the Key image
key_obj = canvas.create_image(100, 100, image=key_img_resized)
canvas.tag_bind(key_obj, '<B1-Motion>', lambda event, id=key_obj: on_drag(event, id))

# Load and display the Rubrics image
rubrics_obj = canvas.create_image(700, 500, image=rubrics_img_resized)

start_moving_image()

# Run the Tkinter loop
root.mainloop()
