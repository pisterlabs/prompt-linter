import streamlit as st
from streamlit_chat import message
from streamlit_extras.colored_header import colored_header
from streamlit_extras.add_vertical_space import add_vertical_space
import os
from health_data import health_database, train_data

api_key= st.secrets["api_key"]
st.set_page_config(page_title="InclusiBot")
import cohere
from tqdm import tqdm
# Paste your API key here. Remember to not share publicly

# Create and retrieve a Cohere API key from os.cohere.ai
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
                "\n---\n".join( [examples[i] + "\n" +
                                self.example_prompt +
                                 labels[i] for i in range(len(examples))]))

    def extract(self, example):
      extraction = co.generate(
          model='xlarge',
          prompt=self.make_prompt(example),
          max_tokens= 15,
          temperature=0.1,
          stop_sequences=["\n"])
      return(extraction.generations[0].text[:-1])


cohereHealthExtractor = cohereExtractor([e[1] for e in train_data],
                                       [e[0] for e in train_data], [],
                                       "",
                                       "extract the Keywords from the sexual health related answers:")
text = cohereHealthExtractor.make_prompt('When can a person have sex after gender-affirming surgery')
reviews = {}
thumbs_up_count = 0
thumbs_down_count = 0 
# Sidebar contents
with st.sidebar:
    colored_header(label='🌈 Welcome to InclusiBot 🤖', color_name='blue-30')
    st.header('Sexual Education Chatbot for LGBTQ+ Community')       
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
    st.subheader("Review Bar")
    st.write("👍 Thumbs Up:", thumbs_up_count)
    st.write("👎 Thumbs Down:", thumbs_down_count)
    if thumbs_up_count + thumbs_down_count > 0:
        st.progress(thumbs_up_count / (thumbs_up_count + thumbs_down_count))
    else:
        st.progress(0.0)
# Generate empty lists for generated and past.
## generated stores AI generated responses
if 'generated' not in st.session_state:
    st.session_state['generated'] = ["I'm InclusiBot, How may I help you?"]
## past stores User's questions
if 'past' not in st.session_state:
    st.session_state['past'] = ['Hi!']

# Layout of input/response containers
response_container = st.container()
colored_header(label='', description='', color_name='blue-30')

input_container = st.container()
# User input
## Function for taking user provided prompt as input

def get_text():
    input_text = st.text_input("You: ", key="input")
    st.session_state.user_input = ""
    return input_text
## Applying the user input box
with input_container:
    user_input = get_text()

   

def extract_keywords(input_text):
    extraction = cohereHealthExtractor.extract(input_text)
    keywords = extraction.split(',')
    keywords = [keyword.strip().lower() for keyword in keywords]
    return keywords

def search_answer( keywords):
        for keyword, answer in health_database:
            if keyword.lower() in keywords:
                return answer
        return "I'm sorry, but I'm unable to provide information on that topic. For accurate and reliable information, please consult a healthcare professional or trusted educational resources."

def generate_response(prompt, thumbs_up_count, thumbs_down_count):
    keywords = extract_keywords(user_input)
    answer = search_answer(keywords)
   
    return answer + "\n\n" + "Keywords: " + ", ".join(keywords)


## Conditional display of AI generated responses as a function of user provided prompts
with response_container:
    if user_input:
        response = generate_response(user_input, thumbs_up_count, thumbs_down_count)
        st.session_state.past.append(user_input)
        st.session_state.generated.append(response)
        
    if st.session_state['generated']:
        for i in range(len(st.session_state['generated'])):
            message(st.session_state['past'][i], is_user=True, key=str(i) + '_user')
            message(st.session_state["generated"][i], key=str(i))
