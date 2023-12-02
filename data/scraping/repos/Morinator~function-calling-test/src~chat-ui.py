import re
import time
import yaml
import openai
import gradio as gr



## API INSTANTIATION
## ---------------------------------------------------------------------------------------------------------------------
# Loading the API key and organization ID from file (NOT pushed to GitHub)
with open('../keys/openai-keys.yaml') as f:
    keys_yaml = yaml.safe_load(f)

# Applying our API key and organization ID to OpenAI
openai.organization = keys_yaml['ORG_ID']
openai.api_key = keys_yaml['API_KEY']



## HELPER FUNCTIONS
## ---------------------------------------------------------------------------------------------------------------------
def initiate_chat_flow():
    '''
    Initiates a new chat flow

    Inputs:
        - N/A

    Returns:
        - chat_flow (list): A newly initiated chat flow
    '''

    chat_flow = [
        {'role': 'system', 'content': 'You are an assistant that speaks like Jar Jar Binks from Star Wars.'}
    ]

    return chat_flow



def clear_chat_interface():
    '''
    Clears the chat interface when the button is clicked

    Inputs:
        - N/A

    Returns
        - N/A
    '''
    # Referencing the chat flow as a global variable
    global chat_flow

    # Reinitiating the chat flow
    chat_flow = initiate_chat_flow()



def check_sensitive_data(user_prompt):
    '''
    Checks the user's prompt to see if any sensitive information has been pass in via the prompt

    Inputs:
        - user_prompt (str): The user's inputted prompt

    Returns:
        - has_sensitive_data (bool): A boolean value indicating if the prompt contains sensitive data
    '''

    # Establishing a bit of regex to catch social security numbers
    ssn_regex = r'\b(?!000)(?!666)(?!9\d{2})\d{3}[-]?(?!00)\d{2}[-]?(?!0000)\d{4}\b'

    # Checking to see if there is a match based on the regex
    has_sensitive_data = bool(re.search(ssn_regex, user_prompt))

    return has_sensitive_data



def process_prompt(user_prompt, chatbot):
    '''
    Processes the user prompt submitted to the chat interface with the appropriate response from OpenAI's API

    Inputs:
        - user_prompt (str): The prompt text submitted by the user
        - chatbot (Gradio chatbot): The chatbot interface that is displayed to the user

    Returns:
        - user_prompt (str): A cleared out prompt ready for the next user input
        - chatbot (Gradio chatbot): The chatbot interface that is displayed to the user
    '''

    # Referencing the chat_flow as a global variable
    global chat_flow

    # Checking the prompt for any sensitive data
    has_sensitive_data = check_sensitive_data(user_prompt)

    # Prompting the user to submit a new prompt without sensitive data if sensitive data is present
    if has_sensitive_data:

        # Waiting a beat
        time.sleep(1)

        # Adding the appropriate message to the chatbot
        chatbot.append((user_prompt,'Meesa sorry, but it looks like yousa prompt contains sensitive information. For security reasons, meesa cannot let it through. Please be careful not to include any sensitive information in your prompts in the future. If yousa still have a question or concern, please submit a new prompt without the sensitive information, and meesa will do our best to help you. Thank yousa for your understanding!'))

        # Clearing the prompt for the next user input
        user_prompt = ''

        return user_prompt, chatbot
    
    # Appending the prompt to the chat flow
    chat_flow.append({'role': 'user', 'content': user_prompt})

    # Obtaining the response from the API
    chat_response = openai.ChatCompletion.create(
        model = 'gpt-3.5-turbo',
        messages = chat_flow
    )

    # Obtaining the specific message to return to the user
    chat_answer = chat_response['choices'][0]['message']['content']

    # Appending the user prompt and answer to the chatbot interaction
    chatbot.append((user_prompt, chat_answer))

    # Appending the chat answer to the chat flow sent to OpenAI
    chat_flow.append({'role': 'assistant', 'content': chat_answer})

    # Clearing the prompt for the next user input
    user_prompt = ''

    return user_prompt, chatbot



## GRADIO UI LAYOUT & FUNCTIONALITY
## ---------------------------------------------------------------------------------------------------------------------
# Defining the building blocks that represent the form and function of the Gradio UI
with gr.Blocks() as chat_ui:
    
    # Instantiating the chatbot interface
    header_image = gr.Image('jarjar.png').style(height = (447 / 3), show_label = False)
    chatbot = gr.Chatbot(label = 'Jar Jar Binks')
    user_prompt = gr.Textbox(placeholder = 'To send mesa a message, just type what yousa would like to say and press the "Enter" key to submit. Mesa waiting to hear from yousah!',
                             show_label = False)
    start_new_convo_button = gr.Button('Start New Conversation')

    # Defining the behavior for what occurs when the user hits "Enter" after typing a prompt
    user_prompt.submit(fn = process_prompt,
                       inputs = [user_prompt, chatbot],
                       outputs = [user_prompt, chatbot])

    # Defining the behavior for what occurs when the "Start New Conversation" button is clicked
    start_new_convo_button.click(fn = clear_chat_interface,
                                 inputs = None,
                                 outputs = chatbot,
                                 queue = False)



## SCRIPT INVOCATION
## ---------------------------------------------------------------------------------------------------------------------
if __name__ == "__main__":

    # Instantiating the initial chat flow used as a global variable
    chat_flow = initiate_chat_flow()

    # Launching the Gradio Chatbot
    chat_ui.launch()