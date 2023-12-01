
import cohere
import streamlit as st
from streamlit_chat import message
from streamlit_extras.colored_header import colored_header
from streamlit_extras.add_vertical_space import add_vertical_space

from health_data import health_database, train_data
from azure.ai.translation.text import TextTranslationClient, TranslatorCredential
from azure.ai.translation.text.models import InputTextItem

# Retrieve API keys from Streamlit secrets
api_key = st.secrets["api_key"]
azure_key = st.secrets["azure_key"]
endpoints = st.secrets["endpoint"]
region = st.secrets["region"]

# Initialize Azure Translator API client
credential = TranslatorCredential(azure_key, region)
text_translator = TextTranslationClient(endpoint=endpoints, credential=credential)


st.set_page_config(page_title="InclusiBot")
# Paste your API key here. Remember to not share publicly
co = cohere.Client(api_key)


class cohereExtractor():
    def __init__(self, examples, example_labels, labels, task_desciption, example_prompt):
        self.examples = examples
        self.example_labels = example_labels
        self.labels = labels
        self.task_desciption = task_desciption
        self.example_prompt = example_prompt

    def make_prompt(self, example):
        examples = self.examples + [example]
        labels = self.example_labels + [""]
        return (self.task_desciption +
                "\n---\n".join([examples[i] + "\n" +
                                self.example_prompt +
                                labels[i] for i in range(len(examples))]))

    def extract(self, example):
        extraction = co.generate(
            model='xlarge',
            prompt=self.make_prompt(example),
            max_tokens=15,
            temperature=0.1,
            stop_sequences=["\n"])
        return (extraction.generations[0].text[:-1])


cohereHealthExtractor = cohereExtractor([e[1] for e in train_data],
                                        [e[0] for e in train_data], [],
                                        "",
                                        "extract the Keywords from the sexual health related answers:")
text = cohereHealthExtractor.make_prompt(
    'When can a person have sex after gender-affirming surgery')
target_language_code = "en" 
# Sidebar contents.
with st.sidebar:
    colored_header(label='🌈 Welcome to InclusiBot 🤖', description= "Here to Help you", color_name='blue-30')
    st.header('Sexual Education Chatbot for LGBTQ+ Community')
    target_language_code = st.text_input("Enter the language code for the response:")

    st.markdown('''
    ## About InclusiBot
    InclusiBot is an AI-powered chatbot designed to provide sexual education information and support for the LGBTQ+ community.
    
    🌈 We believe in inclusivity, respect, and empowerment for all individuals.

    ### How InclusiBot Works
    - InclusiBot utilizes advanced language models to understand your questions and provide accurate responses.
    - It covers various topics related to sexual health, identity, relationships, and more.
    
    💡 Note: InclusiBot is not a substitute for professional advice. Please consult qualified healthcare professionals for specific concerns.
    ''')
    add_vertical_space(5)

# Generate empty lists for generated and past. These will be used to store the chat history
if 'generated' not in st.session_state:
    st.session_state['generated'] = ["I'm InclusiBot, How may I help you?"]
# past stores User's questions
if 'past' not in st.session_state:
    st.session_state['past'] = ['Hi!']

# Layout of input/response containers. These will be used to display the chat history
response_container = st.container()
colored_header(label='', description='', color_name='blue-30')

input_container = st.container()
# User input
# Function for taking user provided prompt as input, and generating a response
def translate_text(text, target_language):
    
    target_languages = [target_language]
    input_text_elements = [ InputTextItem(text = text) ]

    response = text_translator.translate(content = input_text_elements, to = target_languages)
    translation = response[0] if response else None;
    if translation:
        for translated_text in translation.translations:
            return translated_text.text
    else:
        return text

# ...
def get_text():
    input_text = st.text_input("You: ", key="input")
    
    return input_text


# Applying the user input box
with input_container:
    user_input = get_text()


def extract_keywords(input_text):
    extraction = cohereHealthExtractor.extract(input_text)
    keywords = extraction.split(',')
    keywords = [keyword.strip().lower() for keyword in keywords]
    return keywords


def search_answer(keywords):
    for keyword, answer in health_database:
        if keyword.lower() in keywords:
            return answer
    return "I'm sorry, but I'm unable to provide information on that topic. For accurate and reliable information, please consult a healthcare professional or trusted educational resources."


# def generate_response(prompt, thumbs_up_count, thumbs_down_count):
#     keywords = extract_keywords(user_input)
#     answer = search_answer(keywords)

#     return answer + "\n\n" + "Keywords: " + ", ".join(keywords)
def generate_response(prompt, target_language):
    # Detect the language of the prompt
    keywords = extract_keywords(user_input)
    answer = search_answer(keywords)

    translated_answer = translate_text(answer, target_language)

    return translated_answer + "\n\n" + "Keywords: " + ", ".join(keywords)

with response_container:
    if user_input:
        if target_language_code == "":
            target_language_code = "en"
       
        response = generate_response(
            user_input,  target_language_code
        )
        st.session_state.past.append(user_input)
        st.session_state.generated.append(response)
        st.empty()

    if st.session_state['generated']:
        for i in range(len(st.session_state['generated'])):
            message(st.session_state['past'][i], is_user=True, key=str(i) + '_user')
            message(st.session_state["generated"][i], key=str(i))

