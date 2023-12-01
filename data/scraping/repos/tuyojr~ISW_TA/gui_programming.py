from tkinter import *
from tkinter import filedialog as fd
import tkinter.messagebox
from PIL import Image, ImageTk
import time
import wikipedia
from gtts import gTTS
import os
import speech_recognition as sr 
import pyttsx3
import openai

# # this is a GUI module
# # tkinter is an interface to the Tk GUI toolkit

# # Frame is a predefined class in tkinter. It is a container widget which is used to contain other widgets.
# # It works like a container which is responsible for arranging the position of other widgets.
# class Window(Frame):
#     def __init__(self, master = None):
#         Frame.__init__(self, master)
#         self.master = master

# # root is the main window
# # initialize the tkinter window
# root = Tk()

# # object of the frame class
# app = Window(root)

# # set the window title
# root.wm_title("GUI Programming")


# # mainloop() is an infinite loop used to run the application, 
# # wait for an event to occur and process the event as long as the window is not closed
# root.mainloop()

# BUTTON CLASS
# class Window(Frame):
#     def __init__(self, master = None):
#         Frame.__init__(self, master)
#         self.master = master

#         # widget can take all window
#         self.pack(fill=BOTH, expand=1)

#         # create a button instance and link it to the callback function (exit button)
#         exitButton = Button(self, text="Exit", fg='white', bg='gray', command=self.clickExitButton)

#         # place the button at the top left corner
#         # exitButton.place(x=0, y=0)
#         exitButton.pack(side=LEFT)
    
#     def clickExitButton(self):
#         print("Exit button clicked.")
#         exit()

# root = Tk()

# app = Window(root)

# root.wm_title("Tkinter Button")

# root.geometry("320x200")

# root.mainloop()

# def dowork():
#     print("Hello World!")

# def exit_frame():
#     print("Program exited successfully!")
#     exit()

# root = Tk()
# root.wm_title("Tkinter Test")
# root.geometry('320x200')

# f = Frame(root)
# # pack the frame into the root window
# f.pack()

# b = Button(f, text='Say', command=dowork)
# # pack the button into the frame which is inside the root window.
# b.pack()

# b2 = Button(f, text='Exit', fg='white', bg='gray', command=exit_frame)
# b2.pack()

# root.mainloop()


# MENU CLASS
# class Window(Frame):
#     def __init__(self, master=None):
#         Frame.__init__(self, master)
#         self.master = master

#         menu = Menu(self.master)
#         self.master.config(menu=menu)

#         fileMenu = Menu(menu)
#         fileMenu.add_command(label="Item")
#         fileMenu.add_command(label="Exit", command=self.exitProgram)
#         menu.add_cascade(label="File", menu=fileMenu)

#         editMenu = Menu(menu)
#         editMenu.add_command(label="Undo")
#         editMenu.add_command(label="Redo")
#         menu.add_cascade(label="Edit", menu=editMenu)

#         databaseMenu = Menu(menu)
#         databaseMenu.add_command(label="Insert", command=self.insert)
#         databaseMenu.add_command(label="Update", command=self.update)
#         databaseMenu.add_command(label="Create", command=self.create)
#         databaseMenu.add_command(label="Delete", command=self.delete)
#         menu.add_cascade(label="Database", menu=databaseMenu)

#     def exitProgram(self):
#         print("Program exited successfully!")
#         exit()

#     def insert(self):
#         print("Records inserted to database successfully!")
#     def update(self):
#         print("Database record updated successfully!")
#     def create(self):
#         print("Table created in database successfully!")
#     def delete(self):
#         print("Record deleted from database successfully!")

# root = Tk()

# app = Window(root)

# root.wm_title("Menu Window")

# root.mainloop()


# LABEL CLASS
# class Window(Frame):
#     def __init__(self, master=None):
#         Frame.__init__(self, master)
#         self.master = master
#         self.pack(fill=BOTH, expand=1)

#         text = Label(self, text='Just do it')
#         text.place(x=7, y=90)
#         # text.pack()

# root = Tk()
# app = Window(root)
# root.wm_title("Label Window")
# root.geometry("700x600")
# root.mainloop()

# class App(Frame):
#     def __init__(self, master=None):
#         Frame.__init__(self, master)
#         self.master = master

#         self.label = Label(text="", fg='blue', font=("Helvetica", 18))
#         self.label.place(x=50, y=80)

#         self.update_clock()

#     def update_clock(self):
#         now = time.strftime("%H:%M:%S")
#         self.label.config(text=now)
#         self.after(1000, self.update_clock)

# root = Tk()
# app = App(root)
# root.wm_title("Label Class Window")
# root.geometry("200x200")
# root.after(1000, app.update_clock)
# root.mainloop()

# # IMAGE CLASS
# class Window(Frame):
#     def __init__(self, master=None):
#         Frame.__init__(self, master)
#         self.master = master
#         self.pack(fill=BOTH, expand=1)

#         load = Image.open("C:\\Users\\Adedolapo.Olutuyo\\Documents\\ISW_TA\\python\\shisui.jpg")
#         # load = Image.urlopen("https://github.com/tuyojr/ISW_TA/blob/main/python/shisui.jpg")
#         render = ImageTk.PhotoImage(load)
#         img = Label(self, image=render)
#         img.image = render
#         img.place(x=0,y=0)

# root = Tk()
# app = Window(root)
# root.wm_title("Image and Label")
# root.geometry("700x500")
# root.mainloop()

# Tkinter Scale
# SCALE CLASS
# allows you add a scale or a slider to your window. Example is a volume control.
# It has a minimum and maximum value you can define
# window = Tk()
# window.title("Scale Window")
# window.geometry("500x300")

# label = Label(window, bg='white', fg='black', width=20, text='empty')
# label.pack()

# def print_selection(v):
#     label.config(text='you have selected ' + v)

# scale = Scale(window, label='try me', from_=0, to=10, orient=VERTICAL, length=200, showvalue=0, tickinterval=1, resolution=0.01, command=print_selection)
# scale.pack()

# window.mainloop()

# # Tkinter Frame
# # it lets you organize and group widgets
# # 1. pack() - it organizes widgets in blocks before placing them in the parent widget
# # 2. grid() - it organizes widgets in a table-like structure in the parent widget
# def say_hi():
#     print("hello ~ l")

# root = Tk()

# frame1 = Frame(root)
# frame2 = Frame(root)
# root.title("tkinter frame")

# label=Label(frame1, bg="gray", text="Label", justify=LEFT)
# label.pack(side=LEFT)

# hi_there = Button(frame2, bg="cyan", text="say hi", command=say_hi)
# hi_there.pack()

# frame1.pack(padx=1,pady=1)
# frame2.pack(padx=10,pady=10)

# root.mainloop()

# # Tkinter Frame Photo
# # this example adds a photo to the frame

# root = Tk()
# textLabel = Label(root, text="Label", justify=LEFT, padx=10,)
# textLabel.pack(side=LEFT)

# image = Image.open("C:\\Users\\Adedolapo.Olutuyo\\Documents\\ISW_TA\\python\\shisui.jpg")
# photo = ImageTk.PhotoImage(image)
# imgLabel = Label(root, image=photo)
# imgLabel.pack(side=RIGHT)

# mainloop()

# # Tkinter Listbox
# # A listbox shows a list of options. It won't do anything by default.
# # You can link a list box to a function.
# # To add new items, you can use the insert() method.

# window = Tk()
# window.title("Listbox Window")

# window.geometry('500x300')

# var1 = StringVar()
# label = Label(window, bg='gray', fg='black', font=('Arial', 12), width=10, textvariable=var1)
# label.pack()

# def print_selection():
#     value = listbox_object.get(listbox_object.curselection())
#     var1.set(value)

# button1 = Button(window, text='print selection', width=15, height=2, command=print_selection)
# button1.pack()

# var2 = StringVar()
# var2.set((1, 2, 3, 4))
# listbox_object = Listbox(window, listvariable=var2)

# list_items = [11, 22, 33, 44]

# for item in list_items:
#     listbox_object.insert('end', item)
# listbox_object.insert(1, 'first')
# listbox_object.insert(2, 'second')
# listbox_object.delete(2)
# listbox_object.pack()

# window.mainloop()

# # Tkinter Messagebox
# # it is a little popup showing a message. Sometimes it is accompanied by an icon.
# def buttonClick():
#     tkinter.messagebox.showinfo('PopUp', 'SIKE!!!')
#     # tkinter.messagebox.showwarning('title', 'message')
#     # tkinter.messagebox.showerror('title', 'message')

# root = Tk()
# root.title('PopUpBox')
# root.geometry('100x100')
# root.resizable(True, False)
# Button(root, text='click me!', command=buttonClick).pack()
# root.mainloop()

# def callback():
#     name = fd.askopenfilename()
    
#     image = Image.open(name)
#     resized = image.resize((384, 240))
#     photo = ImageTk.PhotoImage(resized)
#     imgLabel = Label(image=photo)
#     imgLabel.image = photo
#     imgLabel.pack()

# error_message = 'Erorr!'
# Button(text='Click to Open File', command=callback).pack(fill=X)
# mainloop()

# # Tkinter Canvas
# root = Tk()

# # create canvas
# myCanvas = Canvas(root, bg='white', height=300, width=300)

# # draw arcs
# coord = 10, 10, 100, 100
# arc = myCanvas.create_arc(coord, start=0, extent=150, fill='gray')
# arc2 = myCanvas.create_arc(coord, start=150, extent=215, fill='cyan')

# line = myCanvas.create_line(300, 300, 10, 10, fill='red')

# # add window to show
# myCanvas.pack()
# root.mainloop()

# # Tkinter Entry
# # It allows users input text into the desktop software.
# # It comes with a label and an input field.

# top = Tk()

# label = Label(top, text='Username')
# label.pack(side=LEFT)

# entry = Entry(top, bd=5)
# entry.pack(side=RIGHT)

# top.mainloop()

# top = Tk()

# label = Label(top, text='Username')
# label.pack(side=LEFT)

# var1 = StringVar()

# entry = Entry(top, bd=5, textvariable=var1)
# entry.pack(side=LEFT)

# def link():
#     print(var1.get())

# button = Button(top, text='Submit', command=link)
# button.pack()

# top.mainloop()

# window = Tk()
# window.title("Entry Window")
# window.geometry('500x300')

# entry1 = Entry(window, show=None, font=('Arial', 14))
# entry2 = Entry(window, show='@', font=('Arial', 14))
# entry1.pack()
# entry2.pack()

# window.mainloop()

# # Tkinter Radiobutton
# # It lets you select from a variety of items. It is similar to a listbox.
# # It only lets you select just one option.
# # You can achieve that by adding the same variable as parameter for radiobuttons.

# window = Tk()
# window.title("Radiobutton Window")
# window.geometry('500x300')

# var = StringVar()
# var.set(' ')
# label = Label(window, bg='white', width=20, text='empty')
# label.pack()

# def print_selection():
#     label.config(text='you have selected ' + var.get())

# radio_button1 = Radiobutton(window, text='Option A', variable=var, value='A')
# radio_button1.pack()
# radio_button2 = Radiobutton(window, text='Option B', variable=var, value='B')
# radio_button2.pack()
# radio_button3 = Radiobutton(window, text='Option C', variable=var, value='C')
# radio_button3.pack()

# button = Button(window, text='print selection', width=13, height=1, command=print_selection)
# button.pack()

# window.mainloop()

# # Tkinter Checkbox
# # It lets you select multiple options. They are like on/off switches and there can be multiple of them.
# # You can achieve that by adding the same variable as parameter for checkboxes.
# window = Tk()
# window.title("Checkbox Window")
# window.geometry('500x300')

# label = Label(window, bg='white', width=20, text='empty')
# label.pack()

# def print_selection():
#     if (var1.get() == 1) & (var2.get() == 0):
#         label.config(text='I love only Python')
#     elif (var1.get() == 0) & (var2.get() == 1):
#         label.config(text='I love only C++')
#     elif (var1.get() == 0) & (var2.get() == 0):
#         label.config(text='I do not love either')
#     else:
#         label.config(text='I love both')

# var1 = IntVar()
# var2 = IntVar()
# checkbox1 = Checkbutton(window, text='Python', variable=var1, onvalue=1, offvalue=0)
# checkbox1.pack()
# checkbox2 = Checkbutton(window, text='C++', variable=var2, onvalue=1, offvalue=0)
# checkbox2.pack()
# button = Button(window, text='print selection', width=13, height=1, command=print_selection)
# button.pack()

# window.mainloop()

# # Tkinter Wikipedia Module
# # The wikipedia module is a multilingual module that lets you search for articles on wikipedia.

# # finding result for a search
# # sentences = 10 refers to the numbers of line
# result = wikipedia.summary("Sherlock Holmes", sentences=10)

# # printing the result
# print(result)

# # Tkinter Google Text to Speech (gTTS) Module
# myTxt = "All the heavens and all the hells are within you!"
# language = 'en'
# myobj = gTTS(text=myTxt, lang=language, slow=False)
# myobj.save("h_and_h.mp3")
# os.system('mediaplayer h_and_h.mp3')

# # combining tkinter, wikipedia, and gTTS
# window = Tk()
# window.title("Wikipedia and gTTS")
# window.geometry('500x300')

# entry = Entry(window, show=None, font=('Arial', 14))
# entry.pack()

# def search():
#     result = wikipedia.summary(entry.get(), sentences=2)
#     language = 'en'
#     resultObj = gTTS(text=result, lang=language, slow=False)
#     print("Search Found!")
#     resultObj.save("result.mp3")
#     os.system('mediaplayer result.mp3')

# button = Button(window, text='Search', width=13, height=1, command=search)
# button.pack()

# window.mainloop()

# # pip install speechrecognition
# # pip install pyaudio

# def recognize_speech():
#     r = sr.Recognizer() #initialize the recognizer
#     with sr.Microphone() as source:
#         audio1 = r.listen(source)
#         mytext = r.recognize_google(audio1)
#         mytext = mytext.lower()
#         print(mytext)

# # Create the main Tkinter window 
# window = Tk() 
# window.title("Speech Recognition") 

# # Create a button to start speech recognition 
# recognize_button = Button(window, text="Recognize Speech", command=recognize_speech) 
# recognize_button.pack() 

# # Start the Tkinter event loop 
# window.mainloop()

# def wiiki():
#     res= wikipedia.summary(recognize_speech())
#     text_entry.insert(END, res)

# def recognize_speech():
#     recognizer = sr.Recognizer()
#     with sr.Microphone() as source:
#         audio = recognizer.listen(source)
#     try:
#         recognized_text = recognizer.recognize_google(audio)
#         text_entry.delete(1.0, END) # Clear previous text
#     except sr.UnknownValueError:
#         text_entry.delete(1.0, END) # Clear previous text
#         text_entry.insert(END, "Speech recognition could not understand audio")

#     except sr.RequestError as e:
#         text_entry.delete(1.0, END) # Clear previous text
#         text_entry.insert(END, f"Could not request results from Google Speech Recognition service; {e}")
#     return recognized_text

# # Create the main Tkinter window
# window = Tk()
# window.title("Speech To Text To Wiki")
# window.geometry('400x400')

# # Create a text entry field
# text_entry = Text(window, height=10, width=50)
# text_entry.pack()

# # Create a button to start speech recognition
# recognize_button = Button(window, text="Recognize Speech", command=wiiki)
# recognize_button.pack()

# window.mainloop()
# # Start the Tkinter event loop

# openai.api_key = "api_key"
# prompt = "The future is now, no?."

# model = "text-davinci-003"

# response = openai.Completion.create(engine=model, prompt=prompt, max_tokens=50)
# generated_text = response.choices[0].text

# print(generated_text)