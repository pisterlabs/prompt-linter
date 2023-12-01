import argparse, json, os, sys
import traceback
from datetime import datetime, date
import tkinter as tk
import tkinter.font
from tkinter import ttk
from promptrix.VolatileMemory import VolatileMemory
from promptrix.FunctionRegistry import FunctionRegistry
from promptrix.GPT3Tokenizer import GPT3Tokenizer
from promptrix.Prompt import Prompt
from promptrix.SystemMessage import SystemMessage
from promptrix.UserMessage import UserMessage
from promptrix.AssistantMessage import AssistantMessage
from promptrix.ConversationHistory import ConversationHistory
from promptrix.promptrixTypes import Message, PromptFunctions, PromptMemory, Tokenizer
from promptrix.GPT3Tokenizer import GPT3Tokenizer
from promptrix.VolatileMemory import VolatileMemory
from alphawave.alphawaveTypes import PromptCompletionOptions, PromptResponse, PromptResponseValidator, Validation
from alphawave.DefaultResponseValidator import DefaultResponseValidator
from alphawave.JSONResponseValidator import JSONResponseValidator
from alphawave.RepairTestClient import TestClient as TestClient
from alphawave.AlphaWave import AlphaWave
from alphawave.OpenAIClient import OpenAIClient
from alphawave.OSClient import OSClient
import alphawave_pyexts.LLMClient as llm
from alphawave_agents.Agent import Agent, AgentOptions
from alphawave_agents.AskCommand import AskCommand
from alphawave_agents.MathCommand import MathCommand
from alphawave_agents.FinalAnswerCommand import FinalAnswerCommand
from alphawave_pyexts.SearchCommand import SearchCommand
import embedding_wv2 as em
import usermodel_class as um

parser = argparse.ArgumentParser()

parser.add_argument('model', type=str, default='wizardLM', choices=llm.get_available_models().append('gpt-4'),help='prompt template model')
args = parser.parse_args()
if args.model is not None:
    model = args.model
else:
    model = 'vicuna_v1.1'

prompt_text = ''

if model.startswith('gpt'):
    client = OpenAIClient(apiKey=os.getenv('OPENAI_API_KEY'), logRequests=True)
else:
    #create OS client
    client = OSClient(apiKey=None)#, logRequests=True)

today_prime = "Todays date is "+date.today().strftime("%b-%d-%Y")+'. The current time is '+datetime.now().strftime('%M minutes after %H hours local time')+'.\n\n'

try:
    em.open()
except Exception as e:
  print(f'Failed to load memory.{str(e)}')

    
def make_agent():
    global client, model, prompt_text, memory
    agent_options = AgentOptions(
        client = client,
        prompt=[prompt_text],
        prompt_options=PromptCompletionOptions(
            completion_type = 'chat',
            model = model,
            temperature = 0.1,
            max_input_tokens = 3600,
            max_tokens = 800,
        ),
        step_delay=0,
        max_steps=30,
        max_repair_attempts=2,
        syntax='JSON',
        tokenizer=GPT3Tokenizer(),
        #logRepairs=True
    )

    # Create an agent
    agent = Agent(options = agent_options)
    agent.addCommand(AskCommand())
    agent.addCommand(FinalAnswerCommand())
    agent.addCommand(MathCommand())
    agent.addCommand(SearchCommand(OpenAIClient(apiKey=os.getenv('OPENAI_API_KEY')), model='gpt-3.5-turbo'))
    memory = agent._options['memory']
    return agent

agent = make_agent()

print(agent._options['prompt_options'].temperature)

def run_query(query):
    global agent
    try:
        response = agent.completeTask(query)
        if type(response) == dict and 'type' in response and response['type'] == 'TaskResponse':
            if response['status'] == 'success' or response['status'] == 'input_needed':
                input_area.insert(tk.END, response['message'])
                #input_area.insert(tk.END, '\n')
    except Exception:
        traceback.print_exc()
        
def show_prompt():
    pass

def set_model():
    pass

def set_max_tokens(maxt):
    agent._options['prompt_options'].max_tokens=maxt

def set_temperature(temp):
    agent._options['prompt_options'].temperature=temp

def set_top_p(top):
    agent._options['prompt_options'].top_p=top

def set_max_tokens(maxt):
    agent._options['prompt_options'].max_tokens=maxt

def set(text):
    global temperature, top_p, max_tknns
    try:
        if 'temp' in text:
            idx = text.find('=')
            if idx > 0:
                temperature = float(text[idx+1:].strip())
        if 'top_p' in text:
            idx = text.find('=')
            if idx > 0:
                top_p = float(text[idx+1:].strip())
        if 'max_tkns' in text:
            idx = text.find('=')
            if idx > 0:
                max_tkns = int(text[idx+1:].strip())
    except Exception as e:
        print ("parse failed", e)

PREV_LEN = 0
def submit():
    global PREV_LEN
    input_text = input_area.get("1.0", tk.END)
    if len(input_text) > PREV_LEN:
        new_text = input_text[PREV_LEN:].strip()
    else:
        new_text = input_text.strip()
    #input_area.insert(tk.END, '\n')


    set_max_tokens(int(max_tkns.get()))
    set_temperature(float(temperature.get()))
    set_top_p(float(top_p.get()))
    response = run_query(new_text)
    input_area.insert(tk.END, '\n')
    PREV_LEN=len(input_area.get("1.0", tk.END))-1
    return(input, response)

def setPrompt():
    global prompt_text, agent
    prompt_text = input_area.get("1.0", tk.END)
    prompt_text = prompt_text.strip()
    if prompt_text.startswith("NoPrompt"):
        prompt_text = ''
    if prompt_text.startswith("Helpfull"):
        prompt_text = 'Respond as a helpful, friendly, information assistant. Say "I don\'t know" if you have no information"'
    if prompt_text.startswith("Bhagaven"):
        prompt_text = 'Respond as a compassionate, empathetic, self-realized follower of Ramana Maharshi'
    if prompt_text.startswith("ACT"):
        prompt_text = 'Respond as a compassionate, empathetic, counselor familiar with Acceptance Commitment Theray'
    if prompt_text.startswith("Flirt"):
        prompt_text = 'Respond as a friendly, chatty, flirty young woman named Samantha. You are 21 years old.'
    agent = make_agent()

def clear():
    global memory, PREV_POS
    PREV_POS="1.0"
    input_area.delete("1.0", tk.END)
    memory.set('history', [])
    
root = tk.Tk()

root.title(args.model)
root.geometry("1440x1280")
#root.config(cursor="watch")
root.rowconfigure(0, weight=1)
root.columnconfigure(0, weight=1)

textFrame = tk.Frame(root, bg='#101820', padx=5, pady=5)
controlFrame = tk.Frame(root, bg="#101010")#, width=2)
textFrame.grid(row=0, column=0, sticky="nsew")
controlFrame.grid(row=0, column=1, sticky="nsew")

searchLevel = 0  # 0 == no search, 1 = normal search, 2 = deep search

def on_value_change(*args):
    print("The selected option is:", float(temperature.get()))
    
# Create a style
style = ttk.Style()

# Modify the label style
style.configure("TLabel", background="grey", foreground="black", font=("Arial", 12))

# Modify the combobox style
style.configure("TCombobox", fieldbackground="grey", foreground="black", selectbackground="grey", font=("Arial", 12))
# Set the font of the dropdown list
style.configure("TCombobox", selectbackground='gray', fieldbackground='black', background='light gray')
style.map('TCombobox', fieldbackground=[('readonly','grey')])
style.configure("TCombobox", font=('Arial', 12))  # sets the font in the entry part
style.configure("TCombobox.Listbox", font=('Arial', 12), background='grey')  # sets the font in the dropdown list part



input_area = tk.Text(textFrame, height=80,bg='#101820', fg='AntiqueWhite1', font=("Bitstream Charter", 11), wrap=tk.WORD)
input_area.grid(row=0, column=0, sticky='nsew')
input_area.pack(fill=tk.BOTH, expand=1)
submit_button = tk.Button(controlFrame,  text="Submit", command=submit, font=("Arial", 12), bg='grey')
submit_button.pack(side='top', fill='x')

clear_button = tk.Button(controlFrame,  text="Clear", command=clear, font=("Arial",12), bg='grey')
clear_button.pack(side='top', fill='x')

temperature = tk.StringVar()
temperature.set('.1')
temperature.trace("w", on_value_change)
temp_label = ttk.Label(controlFrame, text="Temperature", style='TLabel')
temp_label.pack(side='top', fill='x')
temp_button = ttk.Combobox(controlFrame, textvariable=temperature, values = [.01,.1,.2,.4,.5,.7,.9,1.0], style='TCombobox')
temp_button.pack(side='top', fill='x')

top_p = tk.StringVar()
top_p.set('1.0')
top_p_label = ttk.Label(controlFrame, text="Top_p", style='TLabel')
top_p_label.pack(side='top', fill='x')
top_p_button = ttk.Combobox(controlFrame, textvariable=top_p, values = [.1,.2,.3,.4,.5,.6,.7,.8,.9,1.0], style='TCombobox')
top_p_button.pack(side='top', fill='x')

max_tkns = tk.StringVar()
max_tkns.set('250')
max_tok_label = ttk.Label(controlFrame, text="MaxTokens", style='TLabel')
max_tok_label.pack(side='top', fill='x')
max_tok_button = ttk.Combobox(controlFrame, textvariable=max_tkns, values = [50,100,250,500,1000,1500,2000,4000,8000], style='TCombobox')
max_tok_button.pack(side='top', fill='x')

prompt_button = tk.Button(controlFrame, text="setPrompt", command=setPrompt, font=("Arial", 12), bg='grey')
prompt_button.pack(side='top', fill='x')

#model_button = tk.Button(controlFrame, text="gpt-3.5-turbo", command=set_model, font=("Arial", 12), bg='grey')
#model_button.pack(side='top', fill='x')

root.rowconfigure(0, weight=1)
root.columnconfigure(0, weight=1)


root.mainloop()

"""
Find an arithmetic expression over the integers 2, 3, and 6 that evaluates to 12. You must use each integer exactly once. Use the following approach. Create a trial expression. Evaluate the results. If it is false, build hypothesis for generating a better expression, then generate a new trial expression and evaluate again. Do this until the correct result is found. If the result is 12 then the process is over.
"""

"""
Human: you are an instruction-following AI. Follow these instructions: Find an arithmetic expression over the integers 2, 3, and 6 that evaluates to 12. You must use each integer exactly once. Use the following approach.
Loop:
  Generate a trial expression using previously generated hypotheses, if any
  Evaluate the trial expression. 
  Test if the evaluation equals 12?
  If not, build a hypothesis to obtain higher success rate in expression generation.
Untill Test is True.
"""
