import pyaudio
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
import speech_recognition as sr
import threading
import ctypes
from matplotlib.animation import FuncAnimation
import pyttsx3
from deepmultilingualpunctuation import PunctuationModel
import pyautogui
import time
import openai
import os

# setting initial state
state="start"

# setting openaiapi
# openai.api_key = os.environ['OPENAI_KEY']
openai.api_key = os.getenv("OPENAI_KEY")

# Initializing the Punctuator Engine
# model = PunctuationModel()
model="test mode"

#define engine for speech
engine = pyttsx3.init('sapi5')
voices = engine.getProperty('voices')
engine.setProperty('voice', voices[1].id)


# function for code execution in a new thread
def code_exec(prog_lang,*args):
    if prog_lang=='python':
        os.system("code Codes\\codefile.py")
        os.system("python Codes\\codefile.py")
    if prog_lang=='html':
        os.system("code Codes\\codefile.html")
        os.system("start Codes\\codefile.html")
    if prog_lang=='java':
        os.system("code Codes\\codefile.java")
        os.system("javac Codes\\codefile.java")
        os.system("java -classpath Codes\\codefile.class")
    if prog_lang=='c++':
        os.system("code Codes\\codefile.cpp")
        os.system("g++ Codes\\codefile.cpp -o codefile.exe")
        os.system("codefile.exe")
    if prog_lang=='cs':
        os.system("code Codes\\codefile.cs")
        os.system("csc Codes\\codefile.cs")
        os.system("Codes\\codefile.exe")
    if prog_lang=="c":
        os.system("code Codes\\codefile.c")
        os.system("gcc Codes\\codefile.c -o codefile.exe")
        os.system("codefile.exe")


#function for speech
def speak(audio):
    try:
        engine.say(audio)
        engine.runAndWait()
    except:
        pass

# Parameters
CHUNKSIZE = 1024 # number of audio samples per frame
RATE = 44100 # sampling rate in Hz
UPDATE_INTERVAL = 20 # update interval for the plot in ms

# Initialize PyAudio
p = pyaudio.PyAudio()

# Open audio stream
stream = p.open(format=pyaudio.paInt16,
                channels=1,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNKSIZE)

# Initialize plot
fig, ax = plt.subplots(facecolor='black', figsize=(3,1), dpi=100)
plt.axis('off')
line, = ax.plot(np.random.rand(CHUNKSIZE), color='red', linewidth=1)
ax.set_ylim(-1, 1)

# Function to update plot
def update_plot(frame):
    # Read audio from stream
    data = stream.read(CHUNKSIZE, exception_on_overflow=False)
    # Convert byte data to numpy array
    samples = np.frombuffer(data, dtype=np.int16)
    # Normalize samples
    samples = samples / 2**15
    # Update plot
    line.set_ydata(samples)
    return line,

# Create animation
ani = FuncAnimation(fig, update_plot, blit=True, interval=UPDATE_INTERVAL)

# Function to check text for keywords
def check_text(text):
    global state
    if 'activate type' in text:
        state="type"
        speak("mode set to typing")
    if 'activate voice mouse' in text:
        state="voicemouse"
        speak("mode set to voice mouse")
    if 'open' in text or 'run' in text or 'launch' in text:
        text=text.replace("open ","")
        text=text.replace("run ","")
        text=text.replace("launch ","")
        text=text.lower()
        if text=="":
            pass
        else:
            speak("opening "+text)
            pyautogui.press('win')
            pyautogui.write(text)
            time.sleep(1)
            pyautogui.press('enter')
    if 'activate mouse' in text:
        pass
    if 'activate code' in text:
        state="code"
        speak("mode set to coding")
    if 'friday terminate' in text:
        speak("terminating. adios")
        os.abort()

#function to code
def codify(text):
    global state
    if "end coding" in text:
        state="start"
        speak("you have stopped coding")
    else:
        if 'python' in text:
            state="start"
            completion = openai.ChatCompletion.create(
            model="gpt-3.5-turbo", 
            messages = [{"role": "system", "content" : "Answer as concisely as possible. I will be giving you a prompt on what will a code do. Give me the code for it but don't explain how the code works. The code should come as a single output, i.e don't output the code in various parts. If creating functions, always include code for main as well"},
            {"role": "user", "content" : text}]
            )
            print(completion['choices'][0]['message']['content'])
            output=completion['choices'][0]['message']['content']
            output=output.replace("```python","```")
            output=(output.split("```"))[1].split("```")[0]
            with open("Codes\\codefile.py", "w") as f:
                f.write(output)
            threading.Thread(target=code_exec, args=(['python'])).start()
            speak("the code is opened in vscode and running. coding mode ended")
        elif 'html' in text:
            state="start"
            completion = openai.ChatCompletion.create(
            model="gpt-3.5-turbo", 
            messages = [{"role": "system", "content" : "Answer as concisely as possible. I will be giving you a prompt on how a webpage should look like and what will its function be. Give me the code for it but don't explain how the code works. The code should contain css and javscript code so the page is responsive. Ise the <script> and <style> tags instead of creating separate files"},
            {"role": "user", "content" : text}]
            )
            print(completion['choices'][0]['message']['content'])
            html=completion['choices'][0]['message']['content']
            html=html.replace("```html","```")
            html=(html.split("```"))[1].split("```")[0]
            with open("Codes\\codefile.html", "w") as f:
                f.write(output)
            threading.Thread(target=code_exec, args=(['html'])).start()
            speak("the code is opened in vscode and running. coding mode ended")
        elif 'java' in text:
            state="start"
            completion = openai.ChatCompletion.create(
            model="gpt-3.5-turbo", 
            messages = [{"role": "system", "content" : "Answer as concisely as possible. I will be giving you a prompt on what will a code do. Give me the code for it but don't explain how the code works. The code should come as a single output, i.e don't output the code in various parts. If creating functions, always include code for main as well"},
            {"role": "user", "content" : text}]
            )
            print(completion['choices'][0]['message']['content'])
            output=completion['choices'][0]['message']['content']
            output=output.replace("```cpp","```")
            output=(output.split("```"))[1].split("```")[0]
            with open("Codes\\codefile.java", "w") as f:
                f.write(output)
            threading.Thread(target=code_exec, args=(['java'])).start()
            speak("the code is opened in vscode and running. coding mode ended")
        elif 'c plus plus' in text:
            state="start"
            completion = openai.ChatCompletion.create(
            model="gpt-3.5-turbo", 
            messages = [{"role": "system", "content" : "Answer as concisely as possible. I will be giving you a prompt on what will a code do. Give me the code for it but don't explain how the code works. The code should come as a single output, i.e don't output the code in various parts. If creating functions, always include code for main as well"},
            {"role": "user", "content" : text}]
            )
            print(completion['choices'][0]['message']['content'])
            output=completion['choices'][0]['message']['content']
            output=output.replace("```cpp","```")
            output=(output.split("```"))[1].split("```")[0]
            with open("Codes\\codefile.cpp", "w") as f:
                f.write(output)
            threading.Thread(target=code_exec, args=(['c++'])).start()
            speak("the code is opened in vscode and running. coding mode ended")
        elif 'c sharp' in text:
            state="start"
            completion = openai.ChatCompletion.create(
            model="gpt-3.5-turbo", 
            messages = [{"role": "system", "content" : "Answer as concisely as possible. I will be giving you a prompt on what will a code do. Give me the code for it but don't explain how the code works. The code should come as a single output, i.e don't output the code in various parts. If creating functions, always include code for main as well"},
            {"role": "user", "content" : text}]
            )
            print(completion['choices'][0]['message']['content'])
            output=completion['choices'][0]['message']['content']
            output=output.replace("```cs","```")
            output=(output.split("```"))[1].split("```")[0]
            with open("Codes\\codefile.cs", "w") as f:
                f.write(output)
            threading.Thread(target=code_exec, args=(['cs'])).start()
            speak("the code is opened in vscode and running. coding mode ended")
        elif ' c ' in text:
            state="start"
            completion = openai.ChatCompletion.create(
            model="gpt-3.5-turbo", 
            messages = [{"role": "system", "content" : "Answer as concisely as possible. I will be giving you a prompt on what will a code do. Give me the code for it but don't explain how the code works. The code should come as a single output, i.e don't output the code in various parts. If creating functions, always include code for main as well"},
            {"role": "user", "content" : text}]
            )
            print(completion['choices'][0]['message']['content'])
            output=completion['choices'][0]['message']['content']
            output=output.replace("```c","```")
            output=(output.split("```"))[1].split("```")[0]
            with open("Codes\\codefile.c", "w") as f:
                f.write(output)
            threading.Thread(target=code_exec, args=(['c'])).start()
            speak("the code is opened in vscode and running. coding mode ended")
        else:
            speak("i'm sorry, i couldn't understand what you meant. please specify the language you want the code in.")

#function for voice mouse
def voice_mouse(text):
    global state
    if "mode reset" in text:
        state="start"
        speak("you have stopped voice mouse")
    if "left click" in text:
        count=text.count("left click")
        for i in range(count):
            pyautogui.click()
    if "right click" in text:
        count=text.count("right click")
        for i in range(count):
            pyautogui.click(button='right')
    if "scroll up" in text:
        count=text.count("scroll up")
        pyautogui.scroll(count*100)
    if "scroll down" in text:
        count=text.count("scroll down")
        pyautogui.scroll(count*-100)
    if "up" in text:
        count=text.count("up")
        pyautogui.moveRel(0, count*-100, duration=0.2)
    if "down" in text:
        count=text.count("down")
        pyautogui.moveRel(0, count*100, duration=0.2)
    if "left" in text:
        count=text.count("left")
        pyautogui.moveRel(count*-100, 0, duration=0.2)
    if "right" in text:
        count=text.count("right")
        pyautogui.moveRel(count*100, 0, duration=0.2)

# function to type
def type_text(text):
    global state
    if "end typing code confirm" in text:
        text=text.replace("end typing code confirm","")
        state="start"
        speak("you have stopped typing")
    if text=="":
        pass
    else:
        punctuated_text = model.restore_punctuation(text)
        pyautogui.typewrite(punctuated_text,0.1)

# Define a function to recognize speech
def recognize_speech():
    global state
    r = sr.Recognizer()
    while True:
        with sr.Microphone() as source:
            print("Speak now...")
            audio = r.listen(source)
            print("Processing...")
            text=""
        try:
            text = r.recognize_google(audio)
            text=text.lower()
        except sr.UnknownValueError:
            print("Sorry, could not understand audio")
        except sr.RequestError as e:
            print("Could not request results from Google Speech Recognition service; {0}".format(e))
        finally:
            if state=="type":
                threading.Thread(target=type_text, args=(text,)).start()
                print("You said: " + text)
            elif state=="voicemouse":
                threading.Thread(target=voice_mouse, args=(text,)).start()
                print("You said: " + text)
            elif state=="code":
                threading.Thread(target=codify, args=(text,)).start()
                print("You said: " + text)
            else:
                threading.Thread(target=check_text, args=(text,)).start()
                print("You said: " + text)

# Start a new thread for speech recognition
speech_thread = threading.Thread(target=recognize_speech)
speech_thread.start()

# Create tkinter window
root = tk.Tk()
root.overrideredirect(True)
root.geometry("300x100+{}+{}".format(ctypes.windll.user32.GetSystemMetrics(0) - 320, 20))
root.resizable(False, False)
root.attributes("-alpha", 0.6)
root.attributes("-topmost", True)

# Create canvas for plot
canvas = tk.Canvas(root, width=300, height=100, highlightthickness=0)
canvas.pack()

# Embed plot in canvas
plot_widget = FigureCanvasTkAgg(fig, master=canvas)
plot_widget.draw()
plot_widget.get_tk_widget().place(relx=0.5, rely=0.5, anchor="center")

# Start tkinter event loop
root.mainloop()

# Stop and close audio stream
stream.stop_stream()
stream.close()
p.terminate()