import tkinter as tk
from tkinter import *
from PIL import ImageTk, Image
from tkcalendar import Calendar
from tkinter import ttk, filedialog
from tkinter.filedialog import askopenfile
from tkinter import font
import pyaudio
import wave
from pydub import AudioSegment
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from string import punctuation
from collections import Counter
from heapq import nlargest
import speech_recognition as sr
import openai
from tkinter import messagebox

def rec_start():
    CHUNK = 1024
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 44100
    RECORD_SECONDS = int(rec_length_var.get())
    WAVE_OUTPUT_FILENAME = "output.wav"

    audio = pyaudio.PyAudio()

    # start recording
    messagebox.showinfo("Recoding started","Start recording...")
    stream = audio.open(format=FORMAT, channels=CHANNELS,
                        rate=RATE, input=True,
                        frames_per_buffer=CHUNK)

    frames = []

    for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        data = stream.read(CHUNK)
        frames.append(data)

    # stop recording
    messagebox.showinfo("Recording ended","Recording Completed & Save Succesfully!")
    stream.stop_stream()
    stream.close()
    audio.terminate()

    # save the recording to a WAV file
    wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(audio.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()
root = tk.Tk()
root.configure(background='#242124')
root.title("Summarizer using Audio")
root.geometry("1000x600")
rec_length_label = ttk.Label(root, text="Recording length (seconds):")
rec_length_label.pack(side=tk.LEFT, padx=100, pady=20)

rec_length_var = tk.StringVar()
rec_length_var.set("5")
rec_length_entry = ttk.Entry(root, textvariable=rec_length_var)
rec_length_entry.pack(side=tk.LEFT, padx=10, pady=20)

record_button = ttk.Button(root, text="Record", command=rec_start)
record_button.pack(side=tk.LEFT, padx=10, pady=10)
# background_image = PhotoImage(file="bg for python.png")
# background_label = Label(root, image=background_image)
# background_label.place(x=0, y=0, relwidth=1, relheight=1)


def com_as():
   root.destroy()
   import aboutus


def SUT():
   root.destroy()
   import SUT


def extra():
   root.destroy()
   import extra


heading_font = font.Font(family="Arial", weight="bold")
menu_font = font.Font(family="Arial")

menu_bar = tk.Menu(root)
root.config(menu=menu_bar)

main_page = tk.Menu(menu_bar, tearoff=0)
menu_bar.add_cascade(label="Index", menu=main_page)
main_page.add_command(label="Index", command=extra)

about_menu = tk.Menu(menu_bar, tearoff=0)
menu_bar.add_cascade(label="About", menu=about_menu)
about_menu.add_command(label="About", command=com_as)

audio_menu = tk.Menu(menu_bar, tearoff=0)
menu_bar.add_cascade(label="Summarizer using Text", menu=audio_menu)
audio_menu.add_command(label="Summarize Text", command=SUT)

date_label = tk.Label(root, text=" Date Of The Meeting",
                      font="allerta_stencil", bg='#413839', foreground='white', pady=0)
date_label.pack()
date_label.place(relx=0.001, rely=0.01)

cal = Calendar(root, selectmode='day', year=2020, month=5, day=22)
cal.pack(pady=20)
cal.place(relx=0.1, rely=0.075678)


def browse_file():
    global T
    global cbutto
    global Summ
    filename = filedialog.askopenfilename(filetypes=(
        ("WAV files", "*.wav"), ("All files", "*.*")))
    T.delete(0, tk.END)
    T.insert(0, filename)
    cbutto.config(state=tk.DISABLED)
    Summ.delete("1.0", tk.END)
    
def convert_to_text():
    global T
    global cbutto
    global Summ
    audio_file = T.get()
    try:
        r = sr.Recognizer()
        with sr.AudioFile(audio_file) as source:
            audio = r.record(source)
        text = r.recognize_google(audio)
        print("Converted audio is: " + text)
        # define the summary length as a percentage of the input message
        SUMMARY_PERCENTAGE = 0.25
        nlp = spacy.load('en_core_web_sm')
        text = nlp(text)
        # Use set() to eliminate duplicates
        stop_word = list(STOP_WORDS)
        punctuations = list(punctuation)
        stopwords = set(stop_word+punctuations)

        # Use list comprehension for efficiency
        keyword = [token.text for token in text if token.text.lower(
        ) not in stopwords and token.pos_ in ['PROPN', 'ADJ', 'NOUN', 'VERB']]

        freq_word = Counter(keyword)

        # Use variable instead of repeating function call
        max_freq = freq_word.most_common(1)[0][1]

        # Use dictionary comprehension for efficiency
        freq_word = {word: freq / max_freq for word, freq in freq_word.items()}

        # compute the summary length based on the input message length and the summary percentage
        if (len(list(text.sents)) > 2):
            summary_length = int(len(list(text.sents)) * SUMMARY_PERCENTAGE)
        else:
            summary_length = 2
        sent_strength = {}
        for sent in text.sents:
            for word in sent:
                if word.text in freq_word:
                    sent_strength[sent] = sent_strength.get(
                        sent, 0) + freq_word[word.text]
        # filter out duplicate sentences from the top sentences
        summarized_sentences = []
        seen_sentences = set()
        for sentence in nlargest(summary_length, sent_strength, key=sent_strength.get):
            if str(sentence) not in seen_sentences:
                summarized_sentences.append(sentence)
                seen_sentences.add(str(sentence))
        final_sentences = [str(sentence) for sentence in summarized_sentences]
        summary = ' '.join(final_sentences)
        summary = openai.summarise(str(text))

        print("Summary is....")
        print(summary)

    except Exception as e:
        print(e)

    if audio_file:
        Summ.insert(tk.END, summary)
        cbutto.config(state=tk.NORMAL)


def convertnlp():
    global T
    global cbutto
    global Summ
    audio_file = T.get()
    try:
        r = sr.Recognizer()
        with sr.AudioFile(audio_file) as source:
            audio = r.record(source)
        text = r.recognize_google(audio)
        print("Converted audio is: " + text)
        # define the summary length as a percentage of the input message
        SUMMARY_PERCENTAGE = 0.25
        nlp = spacy.load('en_core_web_sm')
        text = nlp(text)
        # Use set() to eliminate duplicates
        stop_word = list(STOP_WORDS)
        punctuations = list(punctuation)
        stopwords = set(stop_word+punctuations)

        # Use list comprehension for efficiency
        keyword = [token.text for token in text if token.text.lower(
        ) not in stopwords and token.pos_ in ['PROPN', 'ADJ', 'NOUN', 'VERB']]

        freq_word = Counter(keyword)

        # Use variable instead of repeating function call
        max_freq = freq_word.most_common(1)[0][1]

        # Use dictionary comprehension for efficiency
        freq_word = {word: freq / max_freq for word, freq in freq_word.items()}

        # compute the summary length based on the input message length and the summary percentage
        if (len(list(text.sents)) > 2):
            summary_length = int(len(list(text.sents)) * SUMMARY_PERCENTAGE)
        else:
            summary_length = 2
        sent_strength = {}
        for sent in text.sents:
            for word in sent:
                if word.text in freq_word:
                    sent_strength[sent] = sent_strength.get(
                        sent, 0) + freq_word[word.text]
        # filter out duplicate sentences from the top sentences
        summarized_sentences = []
        seen_sentences = set()
        for sentence in nlargest(summary_length, sent_strength, key=sent_strength.get):
            if str(sentence) not in seen_sentences:
                summarized_sentences.append(sentence)
                seen_sentences.add(str(sentence))
        final_sentences = [str(sentence) for sentence in summarized_sentences]
        summary = ' '.join(final_sentences)

        print("Summary is....")
        print(summary)

    except Exception as e:
        print(e)

    if audio_file:
        Summ.insert(tk.END, summary)
        cbutto.config(state=tk.NORMAL)


def download_text():
    global Summ
    text = Summ.get("1.0", tk.END)
    filename = filedialog.asksaveasfilename(defaultextension=".txt", filetypes=(
        ("Text files", "*.txt"), ("All files", "*.*")))
    if filename:
        with open(filename, "w") as file:
            file.write(text)
            
def grad_date():
    date.config(text="Selected Date is: " + cal.get_date())


# Add Button and Label
calb = Button(root, text="Get Date", command=grad_date,
              font=65, bg='#C7B4F7', bd=4.5, relief='raise')
calb.pack()
calb.place(relx=0.5, rely=0.2)

date = Label(root, text="", font=75, bg='#C7B4F7', bd=4, relief='raise')
date.pack()
date.place(relx=0.67, rely=0.21)

file_label = Label(root, text="Start recording->",
                   font="allerta_stencil", bg='#242124', foreground='white', pady=0)
file_label.place(relx=0.01, rely=0.4567)

bbutto = ttk.Button(root, text="Browse->", command=browse_file)
bbutto.pack(pady=0)
bbutto.place(relx=0.24, rely=0.47)
# T = Text(root, height=1.3, width=70, bd=2.3, relief='sunken', bg='#F3F0E0')
# T.pack()
# T.place(relx=0.37, rely=0.47)
# T.place(relx=0.37, rely=0.47)
cbutto = tk.Button(root, text="Summarize", command=convertnlp, height=1,
                   width=14, bg='#C7B4F7', bd=4, relief='raise', font=("Arial", 12))
cbutto.pack(pady=20)
cbutto.place(relx=0.01, rely=0.532)
dbutto = tk.Button(root, text="Summarize pro", command=convert_to_text, height=1,
                   width=13, bg='#C7B4F7', bd=4.2, relief='raise', font=("Arial", 12))
dbutto.pack(pady=0)
dbutto.place(relx=0.2, rely=0.53)
Summ = Text(root, height=9, width=105, bd=7, relief='raise', bg='#F3F0E0')
Summ.pack()
Summ.place(relx=0.07, rely=0.62)
T = Text(root, height=9, width=105, bd=7, relief='raise', bg='#F3F0E0')
T.pack()
T.place(relx=0.07, rely=0.62)
root.geometry("1000x600")
cbutto = tk.Button(root, text="Download the summary", command="", height=1,
                   width=20, bg='#C7B4F7', bd=4.5, relief='raise', font=("Arial", 13))
cbutto.pack(pady=20)
cbutto.place(relx=0.77, rely=0.9)


# create a PhotoImage object from the image file


root.mainloop()