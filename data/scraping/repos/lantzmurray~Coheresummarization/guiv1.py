
# This is a GUI for the COhere API
# https://github.com/ParthJadhav/Tkinter-Designer
#UI was generated using the tkinter designer by Parth Jadhav
#API AI Text Summarization intergration by Lantz Murray


from pathlib import Path
import webbrowser
#Import OS
import os


from tkinter import *
#Import COhere API
import cohere
# Explicit imports to satisfy Flake8
#from tkinter import Tk, Canvas, Entry, Text, Button, PhotoImage


OUTPUT_PATH = Path(__file__).parent
#Define Assets Path to be parent folder of this file no matter where the actual file is located
#ASSETS_PATH = OUTPUT_PATH / Path(os.path.dirname(os.path.abspath(__file__)))

ASSETS_PATH = OUTPUT_PATH / Path(r"C:\Users\owner\Cohere Projects\Lantz Murray Summatization App v1\build\assets\frame0")

def callback():
    print("Generate button clicked")
    print(entry_1.get("1.0", "end-1c"))  # Get the API key from the entry field
    print(entry_3.get("1.0", "end-1c"))  # Get the text from the entry field
    #entry_1 is the API key
    #entry_3 is the text to be summarized
    #entry_2 is the summarized text
    
    # Initialize the Cohere client with your API key
    api_key = entry_1.get("1.0", "end-1c")  # Get the API key from the entry field
    #define input_text
    input_text = entry_3.get("1.0", "end-1c")  # Get the text from the entry field
    
    print(api_key)  # Print the API key to the console
    
    entry_2.delete("1.0", "end")  # Clear the existing text in entry_2
    #entry_2.insert(END,"response")  # Insert the summarized text into entry_2
    
    text = input_text
    # Call the summarize method on the Cohere client
    co = cohere.Client(api_key)
    
    response = co.summarize(text=text,
                            model="command-nightly",
                            temperature=0.5)
    
    entry_2.insert(END,response[1])  # Insert the summarized text into entry_2
    
    #entry_2(data.get()) # Print the API key to the console

def relative_to_assets(path: str) -> Path:
    return ASSETS_PATH / Path(path)


    
window = Tk()
window.title("Lantz Murray - Cohere API Text Summarization")
window.geometry("888x500")
window.configure(bg = "#FFFFFF")






canvas = Canvas(
    window,
    bg = "#FFFFFF",
    height = 500,
    width = 888,
    bd = 0,
    highlightthickness = 0,
    relief = "ridge"
)
##
canvas.place(x = 0, y = 0)
image_image_1 = PhotoImage(
    file=relative_to_assets("image_1.png"))
image_1 = canvas.create_image(
    444.0,
    244.0,
    image=image_image_1
)

button_image_1 = PhotoImage(
    file=relative_to_assets("button_1.png"))
button_1 = Button(
    image=button_image_1,
    borderwidth=0,
    highlightthickness=0,
    command=lambda: callback(),
    relief="flat"
)
button_1.place(
    x=721.0,
    y=360.0,
    width=148.0,
    height=103.0
)
button_image_2 = PhotoImage(
    file=relative_to_assets("button_2.png"))
button_2 = Button(
    image=button_image_2,
    borderwidth=0,
    highlightthickness=0,
    #click link to linkedin.com
    command=lambda: webbrowser.open(r"https://www.linkedin.com/in/lantz-murray/") ,
    relief="flat"
)
button_2.place(
    x=0.0,
    y=0.0,
    width=50.0, #width of button
    height=50.0 #height of button   
)
    
    
    
    
entry_image_1 = PhotoImage(
    file=relative_to_assets("entry_1.png"))
entry_bg_1 = canvas.create_image(
    785.5,
    205.5,
    image=entry_image_1
)
entry_1 = Text(
    bd=0,
    bg="#D9D9D9",
    fg="#000716",
    highlightthickness=0
)
entry_1.place(
    x=702.0,
    y=186.0,
    width=167.0,
    height=37.0
)

canvas.create_text(
    695.0,
    126.0,
    anchor="nw",
    text="COHERE API HERE:",
    fill="#FFFFFF",
    font=("Inter", 20 * -1)
)

entry_image_2 = PhotoImage(
    file=relative_to_assets("entry_2.png"))
entry_bg_2 = canvas.create_image(
    356.5,
    414.0,
    image=entry_image_2
)
entry_2 = Text(
    bd=0,
    bg="#D9D9D9",
    fg="#000716",
    highlightthickness=0
)
entry_2.place(
    x=52.0,
    y=365.0,
    width=609.0,
    height=96.0
)

canvas.create_text(
    259.0,
    330.0,
    anchor="nw",
    text="Summarized Text",
    fill="#FFFFFF",
    font=("Inter ExtraBold", 20 * -1)
)

entry_image_3 = PhotoImage(
    file=relative_to_assets("entry_3.png"))
entry_bg_3 = canvas.create_image(
    352.5,
    225.0,
    image=entry_image_3
)
entry_3 = Text(
    bd=0,
    bg="#D9D9D9",
    fg="#000716",
    highlightthickness=0
)
entry_3.place(
    x=52.0,
    y=126.0,
    width=601.0,
    height=196.0
)

canvas.create_text(
    194.0,
    94.0,
    anchor="nw",
    text="Enter Text Here For Summarization ",
    fill="#FFFFFF",
    font=("Inter ExtraBold", 20 * -1)
)

canvas.create_text(
    235.0,
    28.0,
    anchor="nw",
    text="Lantz Murray \nAI Text Summarization ",
    fill="#FFFFFF",
    font=("Inconsolata Regular", 20 * -1)
)


canvas.create_text(
    650.0,
    28.0,
    anchor="nw",
    text="Make Sure your Cohere API key is Valid. \nAlso, make sure text is not empty. \nClick Generate to Summarize Text. \nApp takes about 5-10 seconds to Summarize text \nClick LinkedIn icon to visit my LinkedIn Profile.",
    fill="#FFFFFF",
    font=("Inconsolata Regular", 10 * -1)
)

#entry_1 cannot be empty as default text
#entry_1.insert(END,"Enter valid Cohere API Key Here")
#entry_3 cannot be empty as default text
#entry_3.insert(END,"Enter Text To Be Summarized Here. This cannot be empty.")


window.resizable(False, False)
window.mainloop()
