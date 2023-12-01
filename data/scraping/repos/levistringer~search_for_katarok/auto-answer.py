import openai
from dotenv import load_dotenv, find_dotenv
import os
import panel as pn  # GUI

# Read local .env file
_ = load_dotenv(find_dotenv())  

openai.api_key = os.environ['OPENAI_API_KEY']

# Define the prompt for the ChatGPT Job API
inp = pn.widgets.TextInput( placeholder='Enter text here…')

def get_completion_from_messages(messages, model="gpt-3.5-turbo", temperature=0, max_tokens=500):
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=temperature, 
        max_tokens=max_tokens, 
    )
    return response.choices[0].message["content"]

def collect_messages(debug=True):
    user_input = inp.value_input
    if debug: print(f"User Input = {user_input}")
    if user_input == "":
        return
    inp.value = ''
    global context
    #response, context = process_user_message(user_input, context, utils.get_products_and_category(),debug=True)
    response, context = process_user_message(user_input, context, debug)
    response = f"""Job Proposal is: \n{response}"""
    context.append({'role':'assistant', 'content':f"{response}"})
    panels.append(
        pn.Row('User:', pn.pane.Markdown(user_input, width=600)))
    panels.append(
        pn.Row('Assistant:', pn.pane.Markdown(response.replace('####',''), width=600)))
 
    return pn.Column(*panels)



# Load your resume from a text file
with open('resume.txt', 'r') as file:
    resume = file.read()

panels = [] # collect display

delimiter = '####'  # Nice because it's treated as one token
system_message = "You are a system responding to a a job posting. The user will provide a job description delimted by {delimiter} \n\nThis is my resume: " + resume + "\n\nPlease write a job application based on the experience in my resume."

context = [ {'role':'system', 'content':{system_message}} ]  

inp = pn.widgets.TextInput( placeholder='Enter text here…')
# radio_group = pn.widgets.RadioButtonGroup(
#     # name='Complexity Level', options=['Basic', 'Intermediate', 'Expert'], button_type='success')
button_conversation = pn.widgets.Button(name="Create Proposal", button_type="primary")

interactive_conversation = pn.bind(collect_messages, button_conversation)

dashboard = pn.Column(
    inp,
   
    # pn.Row(radio_group),
    pn.Row(button_conversation),
    pn.panel(interactive_conversation, loading_indicator=True, height=300),
)

dashboard

# response = get_completion_from_messages(messages,max_tokens=200)

# Prompt the user for a job description
# job_description = input("Enter the job description: ")

# while True:
#     # Define system message for ChatGPT API
    

#     messages = [
#             {'role': 'system',
#             'content': system_message},
#             {'role': 'user',
#             'content': f"{delimiter}{job_description}{delimiter}"},
#     ]
#     # Define the prompt for the ChatGPT API
#     # prompt = f"I am applying for the job with the following description:\n{job_description}\n\nThis is my resume: {resume}\n\nPlease write a proposal based on the given information."



#     # Generate a proposal using the ChatGPT API
#     response = openai.ChatCompletion.create(
#         model='gpt-3.5-turbo',
#         messages=messages,
#         max_tokens=500,
#         # n=1,
#         # stop=None,
#         temperature=0.7,
#     )


#     #  response = openai.ChatCompletion.create(

#     # print(response)

#     # Extract the generated proposal from the API response
#     proposal = response.choices[0].message["content"]

#     # Print the generated proposal
#     print("\nGenerated Proposal:")
#     print(proposal)
#     choice = input("\nDo you have any additional questions? (yes/no): ")

#     if choice.lower() != 'yes':
#         break

#     # Prompt for the next job description
#     prompt = "Please provide the next job question:\n"
