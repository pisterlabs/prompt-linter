from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from dotenv import find_dotenv, load_dotenv
import os

from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError
import os

from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)

load_dotenv(find_dotenv())

SLACK_BOT_USER_ID = os.environ["SLACK_BOT_USER_ID"]

# class Chewbacca:
#     def __init__(self, ):
#         #for now, same llm for all purposes
#         self.llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=1)
        
#     def respond_to_mention(self, body):
#         text = body["event"]["text"]

#         mention = f"<@{SLACK_BOT_USER_ID}>"
#         text = text.replace(mention, "").strip()
        
#         info_check = self.check_for_enough_information(text)
        
#         if info_check == 'history':
#             # TODO: get the most recent messages from the user
#             pass
#         elif info_check == 'clarify':
#             # TODO: ask user to provide more information
#             pass
        
#         function = self.decide_function(text)
#         response = self.act(text) #oh boy idk if this is good
#         return response
    

def generic_prompt(user_input):
    chat = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=1)
    
    template = """
    You are a helpful assistant that provides information and assistance to users.

    Your goal is to provide accurate and concise information in a polite and friendly manner.

    Listen carefully to the user's request and provide a clear and concise response.

    If you don't know the answer to a question, apologize to the user.

    Always be polite and friendly.

    Remember to keep your responses concise and to the point, and avoid using technical jargon or complex language that the user may not understand.
    """
    
    system_message_prompt = SystemMessagePromptTemplate.from_template(template)

    human_template = "Here's the input from the user: {user_input}"
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

    chat_prompt = ChatPromptTemplate.from_messages(
        [system_message_prompt, human_message_prompt]
    )

    chain = LLMChain(llm=chat, prompt=chat_prompt)
    response = chain.run(user_input=user_input)

    return response
    

def grammarly_prompt(user_input):
    chat = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=1)

    template = """
    You are a helpful assistant that corrects grammar and spelling mistakes in a given text.

    Your goal is to help the user improve their writing by correcting any errors and suggesting improvements.
    
    If the input is short, simply respond with the corrected text, and then briefly highlight what you changed.
    
    If the input is longer, first summarize the changes you made, and then provide the corrected text.
    
    It is ok to not provide any suggestions if none are relevant.
    
    Always be concise, try to make your point using as few characters as possible.

    """

    system_message_prompt = SystemMessagePromptTemplate.from_template(template)

    human_template = "Here's the text to improve: {user_input}"
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

    chat_prompt = ChatPromptTemplate.from_messages(
        [system_message_prompt, human_message_prompt]
    )

    chain = LLMChain(llm=chat, prompt=chat_prompt)
    response = chain.run(user_input=user_input)

    return response
    
    


def draft_email(user_input, name="Landon"):
    chat = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=1)

    template = """
    
    You are a helpful assistant that drafts an email reply based on an a new email.
    
    Your goal is to help the user quickly create a perfect email reply.
    
    Keep your reply short and to the point and mimic the style of the email so you reply in a similar manner to match the tone.
    
    Start your reply by saying: "Hi {name}, here's a draft for your reply:". And then proceed with the reply on a new line.
    
    Make sure to sign off with {signature}.
    
    """

    signature = f"Kind regards, \n\{name}"
    system_message_prompt = SystemMessagePromptTemplate.from_template(template)

    human_template = "Here's the email to reply to and consider any other comments from the user for reply as well: {user_input}"
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

    chat_prompt = ChatPromptTemplate.from_messages(
        [system_message_prompt, human_message_prompt]
    )

    chain = LLMChain(llm=chat, prompt=chat_prompt)
    response = chain.run(user_input=user_input, signature=signature, name=name)

    return response



### I know there's a better way to do this with langchain and openai's functions point.
# prob good to implement that. but it didnt work out of the box
# tools = [
#     Tool(
#         name = 'grammerly',
#         func = grammarly_prompt,
#         description="Corrects grammar and spelling mistakes in user input."
#     ),
#     Tool(
#         name = 'draft_email',
#         func = draft_email,
#         description="Drafts an email based on the users input and/or a previous email."
#     ),
#     Tool(
#         name = 'generic_prompt',
#         func = generic_prompt,
#         description="Provides information and assistance to users. A generic tool when no other functions seem to fit."
#     ),
# ]

# create a dictionary that maps each of the following functions to a description of what that function does
# these descriptions will be used to match user input to the correct function.
# the descriptions should be unique enough that they can be used to match the user input to the correct function

FUNCTION_DESCRIPTIONS = {
    'grammarly_prompt': "Corrects grammar and spelling mistakes in user input.",
    'draft_email': "Drafts an email reply based on a new email.",
    "generic_prompt": "Provides information and assistance to users. A generic tool when no other functions seem to fit."
}

FUNCTION_MAP = {name: globals()[name] for name in FUNCTION_DESCRIPTIONS.keys()}

from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)

load_dotenv(find_dotenv())

def check_for_enough_information(user_input):
    chat = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=1)

    template = f"""
    You are a helpful assistant that determines if there is enough information in a request to properly respond to it.
    You have three options for a response:
    "enough": meaning you do not need any additional information to respond to the request.
    "history": meaning you need more information from the user's prior chat history with you.
    "clarify": meaning you need more information from the user's prior chat history with you and you need the user to clarify their request.
    Remember, your output must be one of "enough", "history", or "clarify".
    """

    system_message_prompt = SystemMessagePromptTemplate.from_template(template)
    template = "Here is the text that you are analyzing: {user_input}"
    human_message_prompt = HumanMessagePromptTemplate.from_template(template)

    chat_prompt = ChatPromptTemplate.from_messages(
        [system_message_prompt, human_message_prompt]
    )

    chain = LLMChain(llm=chat, prompt=chat_prompt)
    response = chain.run(user_input=user_input)

    response = response.lower()
    
    if response not in ['enough', 'history', 'clarify']:
        return 'enough'
    else:
        return response
    


def decide_function(user_input):
    chat = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=1)

    template = f"""
    You are a helpful assistant that decides which function to use based on user input.
    Your goal is to determine which function is most relevant based on the user input.
    Start by analyzing the user input and determining which function is most relevant.
    Your output will be the name of the function that is most relevant. If you dont think any function is relevant, then output "generic_prompt".
    Remember, your output should just be the name of the function and nothing else (or "generic_prompt" if you dont think any function is relevant)
    The following dictionary pairs the name of each function with a description of what that function does:
    """
    for name, description in FUNCTION_DESCRIPTIONS.items():
        template = template + f"\n{name}: {description}"

    system_message_prompt = SystemMessagePromptTemplate.from_template(template)
    human_message_prompt = HumanMessagePromptTemplate.from_template("{user_input}")

    chat_prompt = ChatPromptTemplate.from_messages(
        [system_message_prompt, human_message_prompt]
    )

    chain = LLMChain(llm=chat, prompt=chat_prompt)
    response = chain.run(user_input = user_input)
    
    response = response.lower()
    
    if response not in FUNCTION_DESCRIPTIONS.keys():
        template = template + f"It is required that your output be just one of the following: {FUNCTION_DESCRIPTIONS.keys()}"
        system_message_prompt = SystemMessagePromptTemplate.from_template(template)
        human_message_prompt = HumanMessagePromptTemplate.from_template("{user_input}")

        chat_prompt = ChatPromptTemplate.from_messages(
            [system_message_prompt, human_message_prompt]
        )

        chain = LLMChain(llm=chat, prompt=chat_prompt)
        response = chain.run(user_input = user_input)
        
        response = response.lower()
        
        if response not in FUNCTION_DESCRIPTIONS.keys():
            raise ValueError(f"Langchain couldnt decide what to do")
    
    print(response)
    return FUNCTION_MAP[response]

def respond(body):
    text = body["event"]["text"]

    mention = f"<@{SLACK_BOT_USER_ID}>"
    text = text.replace(mention, "").strip()
    info_check = check_for_enough_information(text)
    if info_check == 'history':
        print('asking for more history')
        # get the most recent messages from the user
        pass
    elif info_check == 'clarify':
        print('asking to clarify')
        # ask user to provide more information
        pass
    
    function = decide_function(text)
    response = function(text)
    return response




def get_recent_messages(channel_id, count=10):
    """
    Get the most recent messages sent in a Slack channel.

    Args:
        channel_id (str): The ID of the Slack channel to get messages from.
        count (int): The number of messages to retrieve. Defaults to 10.

    Returns:
        list: A list of message dictionaries.
    """
    try:
        # Initialize the Slack client with your bot token
        slack_client = WebClient(token=os.environ["SLACK_BOT_TOKEN"])
        response = slack_client.conversations_history(channel=channel_id, limit=count)
        messages = response["messages"]
        return messages
    except SlackApiError as e:
        print(f"Error: {e}")
        return []
    
    
    
