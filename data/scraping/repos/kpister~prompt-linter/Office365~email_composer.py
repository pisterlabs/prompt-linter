# Import required libraries
from langchain import PromptTemplate
import streamlit as st
from langchain.llms import OpenAI
from dotenv import load_dotenv

# Load the environment variables from the .env file
load_dotenv()

# Define the template for the email conversion task
query_template = query_template = """
    Below is an email that may be unstructured and poorly worded.
    Your goal is to:
    - Format the email properly
    - Convert the input email into the tone specified in curly braces. 
    - Convert the input email into the dialect specified in curly braces.

    Take these examples of different tones as reference:
    - Formal: We went to Hyderabad for the weekend. We have a lot of things to tell you.
    - Informal: Went to Hyderabad for the weekend. Lots to tell you.  

    Below are some examples of words in different dialects:
    - American: Garbage, cookie, green thumb, parking lot, pants, windshield, 
                French Fries, cotton candy, apartment
    - British: Green fingers, car park, trousers, windscreen, chips, candyfloss, 
               flag, rubbish, biscuit

    Example Sentences from each dialect:
    - American: As they strolled through the colorful neighborhood, Sarah asked her 
                friend if he wanted to grab a coffee at the nearby café. The fall 
                foliage was breathtaking, and they enjoyed the pleasant weather, 
                chatting about their weekend plans.
    - British: As they wandered through the picturesque neighbourhood, Sarah asked her 
               friend if he fancied getting a coffee at the nearby café. The autumn 
               leaves were stunning, and they savoured the pleasant weather, chatting 
               about their weekend plans.

    Please start the email with a warm introduction. Add the introduction if you need to.
    
    Below is the email, tone, and dialect:
    TONE: {tone}
    DIALECT: {dialect}
    EMAIL: {email}
    
    YOUR {dialect} RESPONSE:
"""

# Create a PromptTemplate instance to manage the input variables and the template
prompt = PromptTemplate(
    input_variables=["tone", "dialect", "email"],
    template=query_template,
)

# Function to load the Language Model
def loadLanguageModel(api_key_openai):
    llm = OpenAI(temperature=.7)
    return llm

# Set up Streamlit app with Header and Title
st.set_page_config(page_title="Professional Email Writer", page_icon=":robot:")
st.header("Professional Email Writer")

# Create columns for the Streamlit layout
column1, column2 = st.columns(2)

# Display text input for OpenAI API Key
def fetchAPIKey():
    input_text = st.text_input(
        label="OpenAI API Key ",  placeholder="Ex: vk-Cb8un42twmA8tf...", key="openai_api_key_input")
    return input_text

# Get the OpenAI API Key from the user
openai_api_key = fetchAPIKey()

# Display dropdowns for selecting tone and dialect
column1, column2 = st.columns(2)
with column1:
    tone_drop_down = st.selectbox(
        'Which tone would you like your email to have?',
        ('Formal', 'Informal'))

with column2:
    dialect_drop_down = st.selectbox(
        'Which English Dialect would you like?',
        ('American', 'British'))

# Get the user input email text
def getEmail():
    input_text = st.text_area(label="Email Input", label_visibility='collapsed',
                              placeholder="Your Email...", key="input_text")
    return input_text

input_text = getEmail()

# Check if the email exceeds the word limit
if len(input_text.split(" ")) > 700:
    st.write("Maximum limit is 700 words. Please enter a shorter email")
    st.stop()

# Function to update the text box with an example email
def textBoxUpdateWithExample():
    print("in updated")
    st.session_state.input_text = "Vinay I am starts work at yours office from monday"

# Button to show an example email
st.button("*Show an Example*", type='secondary',
          help="Click to see an example of the email you will be converting.", on_click=textBoxUpdateWithExample)
st.markdown("### Your Email:")

# If the user has provided input_text, proceed with email conversion
if input_text:
    # if not openai_api_key:
    #     # If API Key is not provided, show a warning
    #     st.warning(
    #         'Please insert OpenAI API Key. Instructions [here](https://help.openai.com/en/articles/4936850-where-do-i-find-my-secret-api-key)', icon="")
    #     st.stop()
    # Load the Language Model with the provided API Key
    llm = loadLanguageModel(api_key_openai=openai_api_key)
    # Format the email using the PromptTemplate and the Language Model
    prompt_with_email = prompt.format(
        tone=tone_drop_down, dialect=dialect_drop_down, email=input_text)
    formatted_email = llm(prompt_with_email)
    # Display the formatted email
    st.write(formatted_email)