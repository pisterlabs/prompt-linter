import openai # type: ignore
import os

def text_to_ascii(input_text:str) -> str:
    ascii_text = ''
    for character in input_text:
        ascii_text += str(ord(character)) + ' '
    return ascii_text.rstrip()  # remove trailing space

# Check if the environment variable exists
if "OPENAIKEY" in os.environ:
    # If it exists, get its value into a Python variable
    api_key = os.environ["OPENAIKEY"]
else:
    raise ValueError("Please set the OPENAIKEY environment variable")

openai.api_key = api_key

# http://127.0.0.1:8088

# Function to instruct ChatGPT with your custom language
def instruct0(prompt: str) -> str:
    prompt=text_to_ascii(prompt)
    # Make a call to the GPT-3 API with your custom language model
    
    response = openai.Completion.create(
        # engine="text-davinci-002",  # Use the GPT-3 engine
        engine="ggml-gpt4all-j",
        prompt=prompt,
        max_tokens=400,  # Set the maximum number of tokens in the response
        temperature=0.5,  # Controls the randomness of the output (higher value means more randomness)
        n=1,  # Set the number of responses to generate
        stop=None,  # Specify custom stop tokens to stop the response if needed
    ) 

    # Extract and return the generated response
    if response['choices'][0]['text']: # type: ignore
        return response['choices'][0]['text'].strip() # type: ignore
    else:
        return "No response from ChatGPT."
    
def instruct(prompt:str) -> None:
    translated_output = instruct0(prompt)
    print (translated_output)

# Lets forget everything from what we already knew from before of 3script

# between subject and action there is a . (dot)

# instruct("""
# difference between moon and sun
#          """)

# exit(0)


instruct("""
I want to design a programming language called 3script
Let’s start with the basics. Do not assume any influence from other programming languages.
Each instruction starts with '!'

an instruction has subject and action part separated by a dot
after the ! we always have the subject on which we do an action.
         
arguments are on same line as ! and are separated by a space

subject is always lower case and have no spaces
action is always lower case and have no spaces

the subject is always first, the action after

the only understood subjects are vm, contact, circle, vdc, group  
everything else is error
         
if user uses 'virtual machine' as object it means vm

only known actions on vm are start,stop,delete,define,list everything else is error
         
only known actions on contact are delete,define,list everything else is error
         
arguments to the actions are separated by ':'
         
arguments with space inside need '' around
         
An example would be

!vm.start size:big location:brussels

if user uses word create or new as action it means define
         
the names used to point to our subject vm are virtual machine, machine, vm, container

I want to be able to use standard conversational text and you show me examples of the language we have created as example

when I ask something the output from now on is always the code which would be needed to do the action, and I only just show the code
         
if an error I describe the error and show

create a virtual machine with name kristof of size large and in city ghent

         
""")

# create me a virtual machine with name test size big in location gent
# create a virtual machine with name kristof of size large and in city ghent 

# create a group with name testgroup and members kristof, guy and isabelle
# description of group is, we want to celebrate our father         
# voting members for group are isabelle and kristof
     