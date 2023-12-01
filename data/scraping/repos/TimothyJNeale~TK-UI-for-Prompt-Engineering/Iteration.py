'''
A Simple Tkinter interfae for chatGTP
'''
# improt Tkinter
import tkinter as tk
from tkinter import ttk
from tkinter import scrolledtext   # for text box 
from tkinter import filedialog     # for open file dialog

import openai
import os

from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())

openai.api_key  = os.getenv('OPENAI_API_KEY')

# Find the starting directory
START_DIR = os.getcwd()

#----------------------------------  Global Variables  ----------------------------------#
frames = []
prompts = []


#----------------------------------  Procedures  ----------------------------------#

# This helper function will make it easier to use prompts and look at the generated outputs:
def get_completion(prompt, model="gpt-3.5-turbo", temperature=0):
    messages = [{"role": "user", "content": prompt}]
    try:

        response = openai.ChatCompletion.create(
            model=model,
            messages=messages,
            temperature=temperature, # this is the degree of randomness of the model's output
        )
    except Exception as e:
        # Produce error box
        tk.messagebox.showerror(title="Error", message=e)
        return None
    
    return response.choices[0].message["content"]

# Retturn the default prompt for each iteration
def return_default_prompt(index=0):
    if index == 0:
        prompt = f"""Your task is to help a marketing team create a description for a retail website of a product based on a technical fact sheet.

Write a product description based on the information provided in the technical specifications delimited by triple backticks.
"""
    elif index == 1:
        prompt = f"""Your task is to help a marketing team create a description for a retail website of a product based on a technical fact sheet.

Write a product description based on the information provided in the technical specifications delimited by triple backticks.

Use at most 50 words.

"""
    elif index == 2:
        prompt = f"""Your task is to help a marketing team create a description for a retail website of a product based on a technical fact sheet.

Write a product description based on the information provided in the technical specifications delimited by triple backticks.

The description is intended for furniture retailers, so should be technical in nature and focus on the materials the product is constructed from.

Use at most 50 words.

"""
    elif index == 3:
        prompt = f"""Your task is to help a marketing team create a description for a retail website of a product based on a technical fact sheet.

Write a product description based on the information provided in the technical specifications delimited by triple backticks.

The description is intended for furniture retailers, so should be technical in nature and focus on the materials the product is constructed from.

At the end of the description, include every 7-character Product ID in the technical specification.

Use at most 50 words.

"""
    elif index == 4:
        prompt = f"""Your task is to help a marketing team create a description for a retail website of a product based on a technical fact sheet.

Write a product description based on the information provided in the technical specifications delimited by triple backticks.

The description is intended for furniture retailers, so should be technical in nature and focus on the materials the product is constructed from.

At the end of the description, include every 7-character Product ID in the technical specification.

After the description, include a table that gives the product's dimensions. The table should have two columns. In the first column include the name of the dimension. In the second column include the measurements in inches only. Give the table the title 'Product Dimensions'.

Format everything as HTML that can be used in a website. Place the description in a <div> element.

"""

    else:
        prompt = None

    return prompt

def get_factsheet(text=""):
    prompt = f"""OVERVIEW
- Part of a beautiful family of mid-century inspired office furniture, 
including filing cabinets, desks, bookcases, meeting tables, and more.
- Several options of shell color and base finishes.
- Available with plastic back and front upholstery (SWC-100) 
or full upholstery (SWC-110) in 10 fabric and 6 leather options.
- Base finish options are: stainless steel, matte black, 
gloss white, or chrome.
- Chair is available with or without armrests.
- Suitable for home or business settings.
- Qualified for contract use.

CONSTRUCTION
- 5-wheel plastic coated aluminum base.
- Pneumatic chair adjust for easy raise/lower action.

DIMENSIONS
- WIDTH 53 CM | 20.87”
- DEPTH 51 CM | 20.08”
- HEIGHT 80 CM | 31.50”
- SEAT HEIGHT 44 CM | 17.32”
- SEAT DEPTH 41 CM | 16.14”

OPTIONS
- Soft or hard-floor caster options.
- Two choices of seat foam densities: 
 medium (1.8 lb/ft3) or high (2.8 lb/ft3)
- Armless or 8 position PU armrests 

MATERIALS
SHELL BASE GLIDER
- Cast Aluminum with modified nylon PA6/PA66 coating.
- Shell thickness: 10 mm.
SEAT
- HD36 foam

COUNTRY OF ORIGIN
- Italy
"""
    
    return prompt

# Define a function for the generate button
def generate(current_tab=0):
    # Get the prompt text

    promptText = prompts[current_tab].get(1.0, tk.END) + f"Technical specifications: ```{get_factsheet()}'''"

    # Get the response text
    response_text = get_completion(promptText)
    chatGPT_response.insert(tk.END, f"AI:\n{response_text}\n\n")
    

# Define a function for the clear file menu item
def clear_prompt(current_tab=0):
    # Clear the promt box currently selected
    promptText = prompts[current_tab-1]
    if promptText is not None:
        promptText.delete(1.0, tk.END)

#----------------------------------   Setup Tkinter  ----------------------------------#

# Create instance
root = tk.Tk()
root.title("ChatGPT")
#root.resizable(0,0) # disable resizing the GUI
root.geometry("550x680")
root.iconbitmap("01 Nenebiker.ico")


#----------------------------------   Menus  ----------------------------------#

# Add menu bar
menu_bar = tk.Menu(root)
root.config(menu=menu_bar)

# Add file menu
file_menu = tk.Menu(menu_bar, tearoff=0)
file_menu.add_command(label="Settings")
file_menu.add_separator()
file_menu.add_command(label="Exit", command=root.quit)

menu_bar.add_cascade(label="File", menu=file_menu)

#----------------------------------   Tabs  ----------------------------------#

# Create Tab Control
tabControl00 = ttk.Notebook(root)
tabControl00.grid(column=0, row=0, padx=10, pady=10,)
tabControl00.enable_traversal()

#----------------------------------  Fact sheet ----------------------------------#  


# Add a tab for fact sheet
tab00 = ttk.Frame(tabControl00)
tabControl00.add(tab00, text="Fact Sheet", padding=10)

# Createg a container frame to hold widgets on the factsheet tab
factsheet_frame = ttk.LabelFrame(tab00, text="Factsheet")
factsheet_frame.grid(column=0, row=1, padx=10, pady=10)

# Adding text box showing the marking prompt 2 instructions
factsheet = scrolledtext.ScrolledText(factsheet_frame, width=55, height=32, wrap=tk.WORD, bg="lightgray")
factsheet.insert(tk.END, f"{get_factsheet()}\n")
factsheet.configure(state='disabled')
factsheet.grid(column=0, row=0, sticky='WE', padx=10, pady=15)

#----------------------------------  Add a tab for the rest of functionality ----------------------------------# 

# Add a tab for iterations
tab01 = ttk.Frame(tabControl00)
tabControl00.add(tab01, text="Iterations", padding=10)

#----------------------------------  Iterations Tab contol ----------------------------------# 

# Create Tab Control
tabControl = ttk.Notebook(tab01)
tabControl.grid(column=0, row=0, padx=10, pady=10,)
tabControl.enable_traversal()


#----------------------------------  First Iterations Tab  ----------------------------------#  
# Iterate through range cresating a tab for each iteration

for i in range(0, 5):
    # Add a tab for Iteration frame  sheet
    tab = ttk.Frame(tabControl)
    tabControl.add(tab, text=f"Prompt {i+1}", padding=10)

    # We are creating a container frame to hold widgets on the first iteration tab
    iteration_frame = ttk.LabelFrame(tab, text=f"Iteration {i+1}")
    iteration_frame.grid(column=0, row=1, padx=10, pady=10)

    # add frame to list of frames
    frames.append(iteration_frame)

    # Adding a Text box Entry widget
    iteration_prompt = scrolledtext.ScrolledText(iteration_frame, width=50, height=10, wrap=tk.WORD)
    iteration_prompt.insert(tk.END, return_default_prompt(i))
    iteration_prompt.grid(column=0, row=1, sticky='WE', padx=10, pady=10)

    # Add prompt to list of prompts
    prompts.append(iteration_prompt)

#----------------------------------   Action Frame  ----------------------------------#

# We are creating a container frame to hold widgets on the root
action_frame = ttk.LabelFrame(tab01, text="Response")
action_frame.grid(column=0, row=2, padx=10, pady=10)

# Adding a Button call generate passing the prompt entered by the user
action = ttk.Button(action_frame, text="Generate", command=lambda: generate(tabControl.index(tabControl.select())))
action.grid(column=0, row=0, padx=10, pady=10)

# Adding a Button to clear the response box
action = ttk.Button(action_frame, text="Clear", command=lambda: chatGPT_response.delete(1.0, tk.END))
action.grid(column=1, row=0, padx=10, pady=10)

# Add response scrolled text box
chatGPT_response = scrolledtext.ScrolledText(action_frame, width=55, height=12, wrap=tk.WORD)
chatGPT_response.grid(column=0, row=1, sticky='WE', padx=10, pady=10, columnspan=2)



#----------------------------------   Main Loop  ----------------------------------#

# Start the GUI
root.mainloop()
