import os, sys
import openai
import requests
import pytesseract
import tkinter as tk
from PIL import Image
import pyscreenshot as ImageGrab


debug = True
window = tk.Tk()
openai.api_key = 'YOUR_API_KEY'
tesseract_path = r'C:\Program Files\Tesseract-OCR\tesseract'

# Path to tesseract on pc
pytesseract.pytesseract.tesseract_cmd = tesseract_path


class functions:
  
  def imgCheck():
    if not os.path.exists('newProblem.png'):
      problem_photo = 'https://itzcozi.github.io/itzCozi-Hosting/resources/newProblem.png'
      functions.install(problem_photo, os.getcwd(), 'newProblem.png')
    else:
      pass

  def request():
    # Request AI answer
    response = openai.Completion.create(
      model='text-davinci-003',
      prompt=functions.imgToString(),
      temperature=0.1,
      max_tokens=130
    )
    if debug:
      print('Request made')
    return response
  
  def install(URL, destination, name=""):
    # Download and write to file
    file_content = requests.get(URL)
    open(f'{destination}/{name}', 'wb').write(file_content.content)

  def resetAll():
    # Reset everything
    for widget in window.winfo_children():
      widget.destroy()
    if debug:
      print('Window reset')

  def screenShot(event):
    # Find general x,y location for the problem
    x1 = 340
    y1 = 350
    x2 = 1260
    y2 = 750

    im = ImageGrab.grab(bbox=(int(x1), int(y1), int(x2), int(y2)))
    im.save('newProblem.png')
    functions.imgToString()
    functions.resetAll()
    printResponse()
    if debug:
      print('Screenshot taken')

  def imgToString():
    # Image to string
    currentProblem = pytesseract.image_to_string(Image.open('newProblem.png'))
    if debug:
      print('Problem translated to string')
    return currentProblem


def printResponse():
  # Print the response to tinker window (Find a way to input response into the website)
  window.geometry('680x400')
  window.title('Homework AI')
  if debug:
    print('Started')

  problemWidget = tk.Text(window,height=5,width=52,bg='Black',fg='Green')
  clearall = tk.Button(window,text='reset',command=functions.resetAll,bg='Black',fg='Green')
  Txt = tk.Text(window, height=5, width=52, fg='Green', bg='Black')
  Heading = tk.Label(window,text='------ Delta Math Artificial Intelligence ------',bg='Black',fg='Green')
  if debug:
    print('Display initiated')

  # Config text widgets
  Heading.config(font=('Consolas', 16))
  problemWidget.config(font=('Consolas', 12))
  Txt.config(font=('Consolas', 12))
  clearall.config(font=('Consolas', 16))

  # Config packs
  Heading.pack(ipadx=20,ipady=20,anchor=tk.N,fill=tk.X)
  problemWidget.pack(ipadx=110,ipady=20,anchor=tk.CENTER,expand=True,fill=tk.BOTH)
  Txt.pack(ipadx=125,ipady=20,anchor=tk.CENTER,expand=True,fill=tk.BOTH)
  clearall.pack(anchor=tk.CENTER,fill=tk.X)

  # Insert text to widgets
  problemWidget.config(state='normal')
  problemWidget.insert(tk.END, functions.imgToString())
  problemWidget.config(state='disabled')
  Txt.config(state='normal')
  Txt.insert(tk.END, functions.request())
  Txt.config(state='disabled')

  window.bind('p', functions.screenShot)
  window.mainloop()  # Code will run window in a loop


# Load the tkinter window
try:
  functions.imgCheck()
  printResponse()
  sys.exit(0)
except Exception as e:
  print(f'Exited - {e}\n')
  if debug:
    print('! Window Terminated !')
    sys.exit(1)
