from tkinter import *
from tkinter import filedialog

import ttkbootstrap as tb
from ttkbootstrap.scrolled import ScrolledText
from ttkbootstrap.dialogs import Messagebox

import openai
from moviepy.editor import *
import os.path


root = tb.Window(themename="superhero")
# App Title
root.title("A.I. Video Transcriber")
# Set the icon
root.iconbitmap('transcriber_icon.ico')
root.iconbitmap(default='transcriber_icon.ico')
# Size the app
root.geometry('700x450')

# Create some global variables
global VIDEO_FILE_PATH, AUDIO_FILE_PATH

# Open Video File
def open_video():
	global VIDEO_FILE_PATH
	# Open a video file
	my_file = filedialog.askopenfilename(title="Open Video File",
		filetype=(("MP4 Video", ".mp4"), ("All Files", "*.*")))

	# Make sure the user selected a file
	if my_file:
		VIDEO_FILE_PATH = my_file
		# get the file size
		sz = os.path.getsize(my_file)
		# convert to megabytes
		sz = sz / 1000000

		# clear textbox
		my_text.delete(1.0, END)

		# Output to textbox
		my_text.insert(END, f'File To Convert:\n{my_file}\n\nFile Size:\n{sz} mb')

		# Enable the mp3 button
		save_mp3_button.config(state="normal")

# Convert Video To MP3 Audio File
def save_mp3():
	global AUDIO_FILE_PATH
	my_file = filedialog.asksaveasfilename(title="Save MP3 File", 
		filetype=(("MP3 Audio", ".mp3"),), defaultextension=".mp3")
	
	# Check to make sure the user picked a file name to save as
	if my_file:
		AUDIO_FILE_PATH	= my_file

		# Convert to mp3
		MP4ToMP3(VIDEO_FILE_PATH, AUDIO_FILE_PATH)

		# Get the file size
		sz = os.path.getsize(my_file)
		sz = sz / 1000000

		# output to text widget
		my_text.insert(END, f'\n\n\nSaving Audio File As:\n{AUDIO_FILE_PATH}\n\nFile Size:\n{sz} mb')

		# Make sure the file size is less than 25 mb as per OpenAI specs
		if sz <= 25:
			my_text.insert(END, "\n\n\nYour File Is Ready To Transcribe!\nClick The 'Transcribe Video' Button Below...")
			# Enable the transcribe button
			transcribe_button.config(state="normal")
		else:
			my_text.insert(END, "\n\n\nYour File Is Too Large!\nIt Must Be Less Than 25mb in Size...")


# Clear the screen
def clear_screen():
	# Message Box
	my_mb = Messagebox.okcancel("Are You Sure?!", "Clear Transcribed Text")
	if my_mb == "OK":
		# Delete the text
		my_text.delete(1.0, END)

		# Disable The Two Buttons
		save_mp3_button.config(state="disabled")
		transcribe_button.config(state="disabled")


# Transcribe Audio/Video
def transcribe_it():
	# Clear the text box
	my_text.delete(1.0, END)

	# Set our Open AI API Key
	openai.api_key = "YOUR_API_KEY_HERE"
	# Open MP3 Audio file to transcribe
	audio_file = open(AUDIO_FILE_PATH, "rb")
	# Send the file to OpenAI API to transcribe!!!
	transcript = openai.Audio.transcribe("whisper-1", audio_file)

	# output transcript to the text box
	my_text.insert(END, transcript.text)


# Copy text box to clipboard
def copy_it():
	# Clear the clipboard
	root.clipboard_clear()

	# Save to clipboard
	root.clipboard_append(my_text.get(1.0, END))

	# Message box for success
	mb = Messagebox.ok("The Text Has Been Copied To Your Clipboard...", "Copy Complete!")


# Save textbox to text file
def save_text():
	# open a file dialog
	my_file = filedialog.asksaveasfilename(title="Save Transcript",
		filetype=(("Text File .txt", ".txt"),), defaultextension=".txt")

	if my_file:
		# Save file
		text_file = open(my_file, 'w')
		text_file.write(my_text.get(1.0, END))
		text_file.close

		# Create messagebox
		mb = Messagebox.ok("The Text Has Been Saved!", "Save Complete")

# Text box
my_text = ScrolledText(root, height=20, width=110, wrap=WORD, autohide=False)
my_text.pack(pady=15)

# Create a frame
my_frame = Frame(root)
my_frame.pack()

# Create some buttons
open_button = tb.Button(my_frame, bootstyle="light", text="Convert Video To MP3", command=open_video)
open_button.grid(row=0, column=0)

save_mp3_button = tb.Button(my_frame, bootstyle="light", text="Save MP3 As...", state="disabled", command=save_mp3)
save_mp3_button.grid(row=0, column=1, padx=30)

copy_button = tb.Button(my_frame, bootstyle="light", text="Copy Text To Clipboard", command=copy_it)
copy_button.grid(row=0, column=2)

save_button = tb.Button(my_frame, bootstyle="light", text="Save Text", command=save_text)
save_button.grid(row=0, column=3, padx=(30, 0))

clear_button = tb.Button(my_frame, bootstyle="light", text="Clear Screen", command=clear_screen)
clear_button.grid(row=0, column=4, padx=(30, 0))

transcribe_button = tb.Button(root, bootstyle="light", text="Transcribe Video!", width=108, state="disabled", command=transcribe_it)
transcribe_button.pack(pady=20)


# Convert MP4 to MP3 Using MoviePy
def MP4ToMP3(mp4, mp3):
	try:
		FILETOCONVERT = AudioFileClip(mp4)
		FILETOCONVERT.write_audiofile(mp3)
		FILETOCONVERT.close()

	except:
		my_text.insert(END, "\n\nThere was a Problem, Please Try Again...")



root.mainloop()