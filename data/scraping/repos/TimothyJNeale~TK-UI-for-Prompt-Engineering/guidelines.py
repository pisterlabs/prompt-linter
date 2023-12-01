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

# Generate a summarizing prompt
def generate_summarizing_prompt(text):
    prompt = f"""Summarize the text delimited by triple backticks into a single sentence.
    ```{text}```
    """
    return prompt

# Generate a JSON request prompt
def generate_json_prompt(text):
    prompt = f"""Answer the request delimited by triple backticks in a JSON format.
    ```{text}```
    """
    return prompt

# Generate a condition prompt
def generate_condition_prompt(text):
    prompt = f"""You will be provided with text delimited by triple quotes. 
If it contains a sequence of instructions, re-write those instructions in the following format:

    Step 1 - ...
    Step 2 - …
    …
    Step N - …

If the text does not contain a sequence of instructions,then simply write "No steps provided."

    '''{text}'''
    """

    return prompt

# Generate a few-shot prompt
def generate_fewshot_prompt(text):
    prompt = f"""Your task is to answer in a consistent style.

<child>: Teach me about patience.

<grandparent>: The river that carves the deepest valley flows from a modest spring; the grandest symphony originates from a single note; the most intricate tapestry begins with a solitary thread.

<child>: {text}
    
<grandparent>: ...
"""
    return prompt

# Generate a time to think prompt 1
def generate_time1_prompt(text):
    prompt = f"""Perform the following actions: 
1 - Summarize the following text delimited by triple backticks with 1 sentence.
2 - Translate the summary into French.
3 - List each name in the French summary.
4 - Output a json object that contains the following keys: french_summary, num_names.

Separate your answers with line breaks.

Text: ```{text}```"""
    
    return prompt

# Generate a time to think prompt 2
def generate_time2_prompt(text):
    prompt = f"""Your task is to perform the following actions: 
1 - Summarize the following text delimited by triple backticks with 1 sentence.
2 - Translate the summary into French.
3 - List each name in the French summary.
4 - Output a json object that contains the 
  following keys: french_summary, num_names.

Use the following format:
Text: <text to summarize>
Summary: <summary>
Translation: <summary translation>
Names: <list of names in Italian summary>
Output JSON: <json with summary and num_names>

Text: ```{text}```"""
    
    return prompt

# Generate a marking prompt 1
def generate_marking1_prompt(text=""):
    prompt = f"""Determine if the student's solution is correct or not.
Question:
I'm building a solar power installation and I need \
 help working out the financials. 
- Land costs $100 / square foot
- I can buy solar panels for $250 / square foot
- I negotiated a contract for maintenance that will cost \ 
me a flat $100k per year, and an additional $10 / square \
foot
What is the total cost for the first year of operations 
as a function of the number of square feet.

Student's Solution:
Let x be the size of the installation in square feet.
Costs:
1. Land cost: 100x
2. Solar panel cost: 250x
3. Maintenance cost: 100,000 + 100x
Total cost: 100x + 250x + 100,000 + 100x = 450x + 100,000
"""
    
    return prompt

def generate_marking2_prompt(text=""):
    prompt = f"""Your task is to determine if the student's solution \
is correct or not.
To solve the problem do the following:
- First, work out your own solution to the problem. 
- Then compare your solution to the student's solution \ 
and evaluate if the student's solution is correct or not. 
Don't decide if the student's solution is correct until 
you have done the problem yourself.

Use the following format:
Question:
```
question here
```
Student's solution:
```
student's solution here
```
Actual solution:
```
steps to work out the solution and your solution here
```
Is the student's solution the same as actual solution \
just calculated:
```
yes or no
```
Student grade:
```
correct or incorrect
```

Question:
```
I'm building a solar power installation and I need help \
working out the financials. 
- Land costs $100 / square foot
- I can buy solar panels for $250 / square foot
- I negotiated a contract for maintenance that will cost \
me a flat $100k per year, and an additional $10 / square \
foot
What is the total cost for the first year of operations \
as a function of the number of square feet.
``` 
Student's solution:
```
Let x be the size of the installation in square feet.
Costs:
1. Land cost: 100x
2. Solar panel cost: 250x
3. Maintenance cost: 100,000 + 100x
Total cost: 100x + 250x + 100,000 + 100x = 450x + 100,000
```
Actual solution:"""
    
    return prompt

# Find promt text box currently selected
def get_prompt_text_box(current_tab=0):
    if current_tab == 0:
        promptText = summary_prompt
    elif current_tab == 1:
        promptText = json_prompt
    elif current_tab == 2:
        promptText = condition_prompt
    elif current_tab == 3:
        promptText = fewshot_prompt
    elif current_tab == 4:
        promptText = time1_prompt
    elif current_tab == 5:
        promptText = time2_prompt
    elif current_tab == 6:
        promptText = marking1_prompt
    elif current_tab == 7:
        promptText = marking2_prompt
        
    else:
        promptText = None

    return promptText

# Define a function for the generate button
def generate(current_tab=0):

    promptText = get_prompt_text_box(current_tab)
    if current_tab == 0:
        prompt = generate_summarizing_prompt(promptText.get(1.0, tk.END))
        AI = "Summerizing AI"
    elif current_tab == 1:
        prompt = generate_json_prompt(promptText.get(1.0, tk.END))
        AI = "JSON AI"
    elif current_tab == 2:
        prompt = generate_condition_prompt(promptText.get(1.0, tk.END))
        AI = "CONDITION AI"
    elif current_tab == 3:
        prompt = generate_fewshot_prompt(promptText.get(1.0, tk.END))
        AI = "FEWSHOT AI"
    elif current_tab == 4:
        prompt = generate_time1_prompt(promptText.get(1.0, tk.END))
        AI = "TIME TO THINK 1 AI"
    elif current_tab == 5:
        prompt = generate_time2_prompt(promptText.get(1.0, tk.END))
        AI = "TIME TO THINK 2 AI"
    elif current_tab == 6:
        prompt = generate_marking1_prompt(promptText.get(1.0, tk.END))
        AI = "MARKING 1 AI"
    elif current_tab == 7:
        prompt = generate_marking2_prompt(promptText.get(1.0, tk.END))
        AI = "MARKING 2 AI"

    else:
        prompt = None
        AI = "AI"

    # Get the response text
    response_text = get_completion(prompt)
    chatGPT_response.insert(tk.END, f"{AI}:\n{response_text}\n\n")

# Define a function for the open file menu item
def open_file(current_tab=0):
    # Open a file dialog box
    file_name = filedialog.askopenfilename(initialdir = START_DIR, 
                                           title = "Load Prompt", filetypes = (("Text files", "*.txt"), ("all files", "*.*")))
    # Open the file and read the contents
    with open(file_name, 'r',  encoding="utf8") as file:
        # How to read the contents of a file when getting UnicodeDecodeError: 
        # 'charmap' codec can't decode byte 0x9d in position 3931: character maps to <undefined>
        text = file.read()

        # Load the promt box currently selected
        promptText = get_prompt_text_box(current_tab)

        if promptText is not None:
            # Clear the text box and insert the contents of the file
            promptText.delete(1.0, tk.END)
            promptText.insert(tk.END, text)
    
    # Close the file
    file.close()

# Define a function for the save file menu item
def save_file(current_tab=0):
    # Open a file dialog box
    file_name = filedialog.asksaveasfilename(initialdir = START_DIR, 
                                           title = "Save Promt", filetypes = (("Text files", "*.txt"), ("all files", "*.*")))
    # Open the file and write the contents
    with open(file_name+".txt", 'w',  encoding="utf8") as file:
        # Save the promt box currently selected
        promptText = get_prompt_text_box(current_tab)
        if promptText is not None:
            file.write(promptText.get(1.0, tk.END))

    # Close the file
    file.close()

# Define a function for the clear file menu item
def clear_prompt(current_tab=0):
    # Clear the promt box currently selected
    promptText = get_prompt_text_box(current_tab)
    if promptText is not None:
        promptText.delete(1.0, tk.END)

#----------------------------------   Setup Tkinter  ----------------------------------#

# Create instance
root = tk.Tk()
root.title("ChatGPT")
#root.resizable(0,0) # disable resizing the GUI
root.geometry("515x690")
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

# Add prompt menu
prompt_menu = tk.Menu(menu_bar, tearoff=0)
prompt_menu.add_command(label="Load Prompt", command=lambda: open_file(tabControl.index(tabControl.select())))
prompt_menu.add_command(label="Save Prompt", command=lambda: save_file(tabControl.index(tabControl.select())))
prompt_menu.add_separator()
prompt_menu.add_command(label="Clear Prompt", command=lambda: clear_prompt(tabControl.index(tabControl.select())))

menu_bar.add_cascade(label="Prompt", menu=prompt_menu)



#----------------------------------   Tabs  ----------------------------------#

#----------------------------------  Summary Tab  ----------------------------------#

# Create Tab Control
tabControl = ttk.Notebook(root)
tabControl.grid(column=0, row=0, padx=10, pady=10)

# Add summarize tab to the tab control
tab1 = ttk.Frame(tabControl)
tabControl.add(tab1, text="Summarize", padding=10)


# Adding text box showing the summerising instructions
summerising_instructions = generate_summarizing_prompt('TEXT TO BE SUMMARIZED')
summary_preprompt = scrolledtext.ScrolledText(tab1, width=50, height=5, wrap=tk.WORD, bg="lightgray")
summary_preprompt.insert(tk.END, f"{summerising_instructions}\n")
summary_preprompt.configure(state='disabled')
summary_preprompt.grid(column=0, row=0, sticky='WE', padx=10, pady=10)

# We are creating a container frame to hold widgets on the first tab
summary_frame = ttk.LabelFrame(tab1, text="Enter your text to be summarized")
summary_frame.grid(column=0, row=1, padx=10, pady=10)

# Adding a Text box Entry widget
default_prompt = f"""You should express what you want a model to do by providing instructions that are as clear and specific as you can possibly make them. This will guide the model towards the desired output, and reduce the chances of receiving irrelevant or incorrect responses. 

Don't confuse writing a clear prompt with writing a short prompt. In many cases, longer prompts provide more clarity and context for the model, which can lead to more detailed and relevant outputs."""
summary_prompt = scrolledtext.ScrolledText(summary_frame, width=50, height=10, wrap=tk.WORD)
summary_prompt.insert(tk.END, f"{default_prompt}\n")
summary_prompt.grid(column=0, row=1, sticky='WE', padx=10, pady=10)


#----------------------------------  JSON Tab  ----------------------------------#


# Add a JSON tab to the tab control
tab2 = ttk.Frame(tabControl)
tabControl.add(tab2, text="JSON", padding=10)

# Add atext box showing the JSON instructions
json_instructions = generate_json_prompt('PROMPT FOR JASON')
json_preprompt = scrolledtext.ScrolledText(tab2, width=50, height=5, wrap=tk.WORD, bg="lightgray")
json_preprompt.insert(tk.END, f"{json_instructions}\n")
json_preprompt.configure(state='disabled')
json_preprompt.grid(column=0, row=0, sticky='WE', padx=10, pady=10)

# We are creating a container frame to hold widgets on the second tab
json_frame = ttk.LabelFrame(tab2, text="Enter your JSON request")
json_frame.grid(column=0, row=1, padx=10, pady=10)


# Adding a Text box Entry widget
default_prompt = f"""Generate a list of three made-up book titles along with their authors and genres."""
json_prompt = scrolledtext.ScrolledText(json_frame, width=50, height=10, wrap=tk.WORD)
json_prompt.insert(tk.END, f"{default_prompt}\n")
json_prompt.grid(column=0, row=1, sticky='WE', padx=10, pady=10)


#----------------------------------  Conditions Tab  ----------------------------------#

# Add a tab for condition satisfaction
tab3 = ttk.Frame(tabControl)
tabControl.add(tab3, text="Condition", padding=10)

# Adding text box showing the condition instructions
condition_instructions = generate_condition_prompt('PROMPT FOR CONDITION')
condition_preprompt = scrolledtext.ScrolledText(tab3, width=50, height=5, wrap=tk.WORD, bg="lightgray")
condition_preprompt.insert(tk.END, f"{condition_instructions}\n")
condition_preprompt.configure(state='disabled')
condition_preprompt.grid(column=0, row=0, sticky='WE', padx=10, pady=10)


# We are creating a container frame to hold widgets on the third tab
condition_frame = ttk.LabelFrame(tab3, text="Enter your condition")
condition_frame.grid(column=0, row=1, padx=10, pady=10)

# Adding a Text box Entry widget
default_prompt = f"""Making a cup of tea is easy! First, you need to get some water boiling. While that's happening, grab a cup and put a tea bag in it. Once the water is hot enough, 
just pour it over the tea bag. Let it sit for a bit so the tea can steep. After a few minutes, take out the tea bag. 

If you like, you can add some sugar or milk to taste. That's it! You've got yourself a delicious cup of tea to enjoy.
"""

condition_prompt = scrolledtext.ScrolledText(condition_frame, width=50, height=10, wrap=tk.WORD)
condition_prompt.insert(tk.END, f"{default_prompt}\n")
condition_prompt.grid(column=0, row=1, sticky='WE', padx=10, pady=10)


#----------------------------------  Few-shot Tab  ----------------------------------#

# Add a tab for few-shot promting
tab4 = ttk.Frame(tabControl)
tabControl.add(tab4, text="Few-shot", padding=10)

# Adding text box showing the condition instructions
fewshot_instructions = generate_fewshot_prompt('ASK QUESTION')
fewshot_preprompt = scrolledtext.ScrolledText(tab4, width=50, height=5, wrap=tk.WORD, bg="lightgray")
fewshot_preprompt.insert(tk.END, f"{fewshot_instructions}\n")
fewshot_preprompt.configure(state='disabled')
fewshot_preprompt.grid(column=0, row=0, sticky='WE', padx=10, pady=10)


# We are creating a container frame to hold widgets on the fourth tab
fewshot_frame = ttk.LabelFrame(tab4, text="Enter your question")
fewshot_frame.grid(column=0, row=1, padx=10, pady=10)

# Adding a Text box Entry widget
default_prompt = f"""Teach me about resilience."""

fewshot_prompt = scrolledtext.ScrolledText(fewshot_frame, width=50, height=10, wrap=tk.WORD)
fewshot_prompt.insert(tk.END, f"{default_prompt}\n")
fewshot_prompt.grid(column=0, row=1, sticky='WE', padx=10, pady=10)


#----------------------------------  Time to think Tab 1 ----------------------------------#

# Add a tab for time to think 2
tab5 = ttk.Frame(tabControl)
tabControl.add(tab5, text="Time 1", padding=10)

# Adding text box showing the condition instructions
time1_instructions = generate_time1_prompt('ADD TEXT')
time1_preprompt = scrolledtext.ScrolledText(tab5, width=50, height=5, wrap=tk.WORD, bg="lightgray")
time1_preprompt.insert(tk.END, f"{time1_instructions}\n")
time1_preprompt.configure(state='disabled')
time1_preprompt.grid(column=0, row=0, sticky='WE', padx=10, pady=10)


# We are creating a container frame to hold widgets on the fith tab
time1_frame = ttk.LabelFrame(tab5, text="Enter text")
time1_frame.grid(column=0, row=1, padx=10, pady=10)

# Adding a Text box Entry widget
default_prompt = f"""In a charming village, siblings Jack and Jill set out on a quest to fetch water from a hilltop well. As they climbed, singing joyfully, misfortune struck—Jack tripped on a stone and tumbled down the hill, with Jill following suit. Though slightly battered, the pair returned home to comforting embraces. Despite the mishap, their adventurous spirits remained undimmed, and they continued exploring with delight."""

time1_prompt = scrolledtext.ScrolledText(time1_frame, width=50, height=10, wrap=tk.WORD)
time1_prompt.insert(tk.END, f"{default_prompt}\n")
time1_prompt.grid(column=0, row=1, sticky='WE', padx=10, pady=10)



#----------------------------------  Time to think Tab 2 ----------------------------------#

# Add a tab for time to think 2
tab6 = ttk.Frame(tabControl)
tabControl.add(tab6, text="Time 2", padding=10)

# Adding text box showing the condition instructions
time2_instructions = generate_time2_prompt('ADD TEXT')
time2_preprompt = scrolledtext.ScrolledText(tab6, width=50, height=5, wrap=tk.WORD, bg="lightgray")
time2_preprompt.insert(tk.END, f"{time2_instructions}\n")
time2_preprompt.configure(state='disabled')
time2_preprompt.grid(column=0, row=0, sticky='WE', padx=10, pady=10)


# We are creating a container frame to hold widgets on the sixth tab
time2_frame = ttk.LabelFrame(tab6, text="Enter text")
time2_frame.grid(column=0, row=1, padx=10, pady=10)

# Adding a Text box Entry widget
default_prompt = f"""In a charming village, siblings Jack and Jill set out on a quest to fetch water from a hilltop well. As they climbed, singing joyfully, misfortune struck—Jack tripped on a stone and tumbled down the hill, with Jill following suit. Though slightly battered, the pair returned home to comforting embraces. Despite the mishap, their adventurous spirits remained undimmed, and they continued exploring with delight."""

time2_prompt = scrolledtext.ScrolledText(time2_frame, width=50, height=10, wrap=tk.WORD)
time2_prompt.insert(tk.END, f"{default_prompt}\n")
time2_prompt.grid(column=0, row=1, sticky='WE', padx=10, pady=10)


#----------------------------------  Marking Tab 1 ----------------------------------#

# Add a tab for student marking 1
tab7 = ttk.Frame(tabControl)
tabControl.add(tab7, text="Marking 1", padding=10)

# Adding text box showing the marking prompt 1 instructions
marking1_instructions = generate_marking1_prompt()
marking1_preprompt = scrolledtext.ScrolledText(tab7, width=56, height=19, wrap=tk.WORD, bg="lightgray")
marking1_preprompt.insert(tk.END, f"{marking1_instructions}\n")
marking1_preprompt.configure(state='disabled')
marking1_preprompt.grid(column=0, row=0, sticky='WE', padx=10, pady=10)


# We are creating a container frame to hold widgets on the seventh tab
marking1_frame = ttk.LabelFrame(tab7, text="Enter text")
marking1_frame.grid(column=0, row=1, padx=10, pady=10)

marking1_prompt = scrolledtext.ScrolledText(marking1_frame, width=50, height=1, wrap=tk.WORD)
marking1_prompt.insert(tk.END, f"")
marking1_prompt.grid(column=0, row=1, sticky='WE', padx=10, pady=10)

# set fram to invisable but still present
marking1_frame.grid_remove()

#----------------------------------  Marking Tab 2 ----------------------------------#  


# Add a tab for student marking 2
tab8 = ttk.Frame(tabControl)
tabControl.add(tab8, text="Marking 2", padding=10)

# Adding text box showing the marking prompt 2 instructions
marking2_instructions = generate_marking2_prompt()
marking2_preprompt = scrolledtext.ScrolledText(tab8, width=56, height=19, wrap=tk.WORD, bg="lightgray")
marking2_preprompt.insert(tk.END, f"{marking2_instructions}\n")
marking2_preprompt.configure(state='disabled')
marking2_preprompt.grid(column=0, row=0, sticky='WE', padx=10, pady=10)


# We are creating a container frame to hold widgets on the eighth tab
marking2_frame = ttk.LabelFrame(tab8, text="Enter text")
marking2_frame.grid(column=0, row=1, padx=10, pady=10)

marking2_prompt = scrolledtext.ScrolledText(marking2_frame, width=50, height=1, wrap=tk.WORD)
marking2_prompt.insert(tk.END, f"")
marking2_prompt.grid(column=0, row=1, sticky='WE', padx=10, pady=10)

# set fram to invisable but still present
marking2_frame.grid_remove()

#----------------------------------   Action Frame  ----------------------------------#

# We are creating a container frame to hold widgets on the root
action_frame = ttk.LabelFrame(root, text="Response", width=400, height=100)
action_frame.grid(column=0, row=2, padx=10, pady=10)

# Adding a Button call generate passing the prompt entered by the user
action = ttk.Button(action_frame, text="Generate", command=lambda: generate(tabControl.index(tabControl.select())))
action.grid(column=0, row=0, padx=10, pady=10)

# Adding a Button to clear the response box
action = ttk.Button(action_frame, text="Clear", command=lambda: chatGPT_response.delete(1.0, tk.END))
action.grid(column=1, row=0, padx=10, pady=10)

# Add response scrolled text box
chatGPT_response = scrolledtext.ScrolledText(action_frame, width=55, height=10, wrap=tk.WORD)
chatGPT_response.grid(column=0, row=1, sticky='WE', padx=10, pady=10, columnspan=2)



#----------------------------------   Main Loop  ----------------------------------#

# Start the GUI
root.mainloop()
