# Import necessary libraries
import streamlit as st
import os
import tempfile
import openai
import sounddevice as sd
import soundfile as sf

from dotenv import load_dotenv
from pynput import keyboard
from PyPDF2 import PdfReader
from elevenlabs import voices, generate, set_api_key, play, save, stream

from streamlit_chat import message

from langchain.prompts import PromptTemplate
from langchain.agents import Tool, AgentType, initialize_agent, load_tools
from langchain.memory import ConversationBufferMemory
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.utilities import SerpAPIWrapper
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.chains import LLMChain
from langchain.callbacks import get_openai_callback

import guidance
import re

# Set recording parameters
DURATION = 3   # duration of each recording in seconds
FS = 44100  # sample rate
CHANNELS = 1  # number of channels

# Create agent with specified model, tools, and temperature
@st.cache_resource
def create_agent(model_name, selected_tools, temperature):
    load_dotenv()
    xiapi = os.getenv('XILABS_API_KEY')
    set_api_key(xiapi)
    memory = create_memory()
    llm = OpenAI(temperature=temperature, model_name=model_name)
    tools = load_tools(selected_tools, llm=llm)
    agent = initialize_agent(tools=tools, llm=llm, agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION, verbose=True, memory=memory)
    return agent

# Create memory for agent
def create_memory():
    memory = ConversationBufferMemory(memory_key="chat_history")

# Record audio with given parameters
def record_audio(duration, fs, channels):
    st.write("Recording...")
    recording = sd.rec(int(duration * fs), samplerate=fs, channels=channels)
    sd.wait()
    st.write("Finished recording.")
    return recording

# Transcribe audio to text using OpenAI
def transcribe_audio(recording, fs):
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_audio:
        sf.write(temp_audio.name, recording, fs)
        temp_audio.close()
        with open(temp_audio.name, "rb") as audio_file:
            transcript = openai.Audio.transcribe("whisper-1", audio_file, prompt="Hello, This transcription comes from a user who is interacting with an Artificial Intelligence assistant.")
        os.remove(temp_audio.name)
    return transcript["text"].strip()

# Generate and play audio from text
def play_generated_audio(text, voice="Adam", model="eleven_monolingual_v1"):
    audio = generate(text=text, voice=voice, model=model,stream=True)
    stream(audio)

# Voice assistant function
def voice_assistant():
    message = ""

    # UI elements
    st.title('Voice Assistant 💬🎙')
   # st.image('voice_assistant_image.jpg', use_column_width=True)  Add an image (replace 'voice_assistant_image.jpg' with your image file)

    st.markdown("""
    Welcome to the Voice Assistant! This application allows you to interact with a voice assistant using your voice. 
    Specify the duration of the recording, choose the model for the assistant, and press the 'Start Recording' button to begin. You can view the conversation history and clear it using the 'Clear Conversation' button.
    """)

    # Sidebar settings
    st.sidebar.title('Settings')
    model_name = st.sidebar.selectbox('Choose the model for the assistant', ['gpt-3.5-turbo', 'gpt-4'], help="GPT-4 is a better model but slower.")
    duration = st.sidebar.number_input('Recording duration in seconds', min_value=1, value=3)
    with st.sidebar.expander('Advanced Settings'):
        temperature = st.slider('Temperature', min_value=0.0, max_value=2.0, value=0.5)
        available_tools = ["python_repl", "requests_all", "terminal", "wolfram-alpha", "ddg-search", "wikipedia", "arxiv", "llm-math", "human", "pal-math", "pal-colored-objects","serpapi"]
        selected_tools = st.multiselect('Choose the tools to load', available_tools, default=["serpapi","wolfram-alpha"], help="These are the tools which the agent will be able to use to help you")

    # Initialize agent
    agent = create_agent(model_name,selected_tools,temperature)

    # Record, transcribe, and generate assistant's message
    col1, col2 = st.columns(2)
    if st.button('Start Recording'):
        if 'chat_log' not in st.session_state:
            st.session_state['chat_log'] = []

        with col1:
            recorded_audio = record_audio(duration, FS, CHANNELS)
            try:
                message = transcribe_audio(recorded_audio, FS)
                st.session_state['chat_log'].append(("You", message))
            except Exception as e:
                st.write("Error in transcription: ", str(e))

        with col2:
            input_dict = {
                'chat_history': st.session_state['chat_log'],
                'input': message
            }
            assistant_message = agent.run(input_dict)
            st.session_state['chat_log'].append(("Assistant", assistant_message))
            play_generated_audio(assistant_message["Assistant"])
            

    # Display chat log
    if 'chat_log' not in st.session_state:
        st.session_state['chat_log'] = []

    for speaker, message in st.session_state['chat_log']:
        st.write(f"{speaker}: {message}")


    # Clear conversation button
    if st.button('Clear Conversation'):
        st.session_state['chat_log'] = []
        agent = create_agent(model_name=model_name, selected_tools=selected_tools,temperature=temperature)

# Ask your PDF function
def ask_your_pdf():
    load_dotenv()
    st.header("Ask your PDF💬")

    # UI elements
    pdf = st.file_uploader("Upload your PDF", type = "pdf")
    key = os.getenv("OPENAI_API_KEY")

    # Sidebar settings
    st.sidebar.title('Settings')

    model_name = st.sidebar.selectbox('Choose the model for Ask Your PDF', ['gpt-4','gpt-3.5-turbo'], help="GPT-4 is a better model but slower.")

    with st.sidebar.expander('Advanced Settings'):
        temperature = st.slider('Temperature', min_value=0.0, max_value=2.0, value=0.5)

    if pdf is not None:
        pdf_reader = PdfReader(pdf)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        
        #split into chunks
        text_splitter = CharacterTextSplitter(
            separator = "\n",
            chunk_size = 1000,
            chunk_overlap = 200,
            length_function = len
        )
        chunks = text_splitter.split_text(text)

        # create embeddings
        embeddings = OpenAIEmbeddings()
        knowledge_base = FAISS.from_texts(chunks,embeddings)

        user_question = st.text_input("Ask a question about your PDF: ")

        # Check if user has entered a question
        if user_question:
            # Search for similar documents in the knowledge base
            docs = knowledge_base.similarity_search(user_question)

            # Initialize the language model with the specified settings
            llm = OpenAI(model_name=model_name, openai_api_key=key, temperature=temperature)

            # Load the question-answering chain
            chain = load_qa_chain(llm, chain_type="stuff")

            # Run the chain with the input documents and user's question
            response = chain.run(input_documents=docs, question=user_question)

            # Display the response
            st.write(response)

            # Play the response as audio if the user clicks the button
            if st.button('Read it to me'):
                play_generated_audio(response)

def dall_e_generator():
    load_dotenv()
    st.title('DALL·E Generator 🎨')
    st.sidebar.title('Settings')
    
    model_name = st.sidebar.selectbox('Choose the model for the chain', ['gpt-4','gpt-3.5-turbo'], help="GPT-4 is a better model but slower.")
    with st.sidebar.expander('Advanced Settings'):
        temperature = st.slider('Temperature', min_value=0.0, max_value=2.0, value=0.5)
    st.markdown("""
    Welcome to the DALL·E Generator! This application allows you to generate images using OpenAI's DALL·E model. 
    Enter a prompt and specify the number of images to generate, then press the 'Generate Images' button to start.
    """)

    # Let the user specify the prompt and number of images to generate
    pr = st.text_input('Enter a prompt for the image generation')
    num_images = st.number_input('Number of images to generate', min_value=1, max_value=10, value=1)

    prompt = PromptTemplate(
        input_variables=["pr"],
        template=f"Given the following prompt: {{pr}}, generate a detailed and creative prompt for DALL-E to create an image. Remember to include as many specific details as possible to ensure a high-quality result from DALL-E. This prompt should be less than 100 words"
    )
    llm = OpenAI(model_name=model_name, temperature=temperature)
    chain = LLMChain(llm=llm, prompt=prompt)
    if num_images>1:
        singleshot = not st.checkbox('Regenerate prompt every iteration', value=True, help="If this box is checked then a new prompt will be generated by the Prompt Generation Chain for every single image. If the box is not checked, only one prompt will be generated by the chain")
    else:
        singleshot = True

    if st.button('Generate Images'):
        if singleshot:
            sendIn = chain.run(pr=pr)
            st.write("Prompt sent in: ", sendIn)
            # Generate images using DALL·E
            response = openai.Image.create(
                prompt=sendIn,
                n=num_images,
                size="1024x1024"
            )
            for i, data in enumerate(response['data']):
                st.image(data['url'], caption='pr')
        else:
            for i in range(num_images):
                sendIn = chain.run(pr=pr)
                st.write("Prompt sent in: ", sendIn)
            # Generate images using DALL·E
                response = openai.Image.create(
                    prompt=sendIn,
                    n=1,
                    size="1024x1024"
            )


            # Display the generated images
                for x, data in enumerate(response['data']):
                    st.image(data['url'], caption='pr')


def run_plan_maker():
    st.sidebar.title('Settings')
    model_name = st.sidebar.selectbox('Choose the model for the chain', [
                                      'gpt-4', 'gpt-3.5-turbo'], help="GPT-4 is a better model but slower.")

    # UI elements
    st.title('Plan Maker 📝🤖')
    # st.image('plan_maker_image.jpg', use_column_width=True)  # Add an image (replace 'plan_maker_image.jpg' with your image file)

    st.markdown("""
    Welcome to the Plan Maker! This application is powered by Microsoft's guidance framework and allows you to generate a plan for any topic. 
    Simply enter the thing you want a plan for, and our AI will generate a plan for you. You can view the generated plan and start a new one at any time.
    """)

    st.text_input("What do you want a plan for?", key="Plan_subject")

    guidance.llm = guidance.llms.OpenAI(model_name)

    def parse_best_plan(prosandcons, options):
        best = re.search(r'Best=(\d+)', prosandcons)
        if not best:
            best = re.search(r'Best.*?(\d+)', 'Best= option is 3')
        if best:
            best = int(best.group(1))
        else:
            best = 0
        return options[best]

    create_plan = guidance('''
        {{#system~}}
        You are a helpful assistant.
        {{~/system}}


        {{#user~}}
        I want to {{goal}}.
        {{~! generate potential options ~}}
        Can you please generate one option for how to accomplish this?
        Please make the option very short, at most one line.
        {{~/user}}

        {{#assistant~}}
        {{gen 'options' n=5 temperature=1.0 max_tokens=600}}
        {{~/assistant}}


        {{#user~}}
        I want to {{goal}}.

        Can you please comment on the pros and cons of each of the following options, and then pick the best option?
        ---{{#each options}}
        Option {{@index}}: {{this}}{{/each}}
        ---
        Please discuss each option very briefly (one line for pros, one for cons), and end by saying Best=X, where X is the best option.
        {{~/user}}

        {{#assistant~}}
        {{gen 'prosandcons' temperature=0.0 max_tokens=600}}
        {{~/assistant}}

        {{#user~}}
        I want to {{goal}}.
        Here is my plan:
        {{parse_best prosandcons options}}
        Please elaborate on this plan, and tell me how to best accomplish it.
        {{~/user}}
        {{#assistant~}}
        {{gen 'plan' max_tokens=500}}
        {{~/assistant}}''')
    if st.session_state.Plan_subject is not None:
        out = create_plan(goal=st.session_state.Plan_subject,
                          parse_best=parse_best_plan)
        st.write(out['plan'])

def expertAdvice():
    st.sidebar.title('Settings')
    model_name = st.sidebar.selectbox('Choose the model for the chain', [
                                      'gpt-4', 'gpt-3.5-turbo'], help="GPT-4 is a better model but slower.")

    # UI elements
    st.title('Expert Advice 💻🧠')
    # st.image('plan_maker_image.jpg', use_column_width=True)  # Add an image (replace 'plan_maker_image.jpg' with your image file)

    st.markdown("""
    Welcome to the Expert Advice! This application is powered by Microsoft's guidance framework and allows you to get expert advice on any topic. 
    Simply enter the thing you want expert advice about, and our AI will choose experts for you and have them work together to answer the question.
    """)


    expert_advice = guidance(
        '''
        {{#system~}}
        You are a marketing director at a large .
        {{~/system}}


        {{#user~}}
        I want a response to the following question:
        {{query}}
        Who are 3 world-class experts (past or present) who would be great at answering this?
        Please don't answer the question or comment on it yet.
        {{~/user}}

        {{#assistant~}}
        {{gen 'experts' temperature=0 max_tokens=300}}
        {{~/assistant}}

        {{#user~}}
        Great, now please answer the question as if these experts had collaborated in writing a joint anonymous answer.
        In other words, their identity is not revealed, nor is the fact that there is a panel of experts answering the question.
        If the experts would disagree, just present their different positions as alternatives in the answer itself (e.g. 'some might argue... others might argue...').
        Please start your answer with ANSWER:
        {{~/user}}

        {{#assistant~}}
        {{gen 'answer' temperature=0 max_tokens=500}}
        {{~/assistant}}''')

    guidance.llm = guidance.llms.OpenAI(model_name)
    st.text_input("What do you want to ask the experts? ", key='eq')
    if st.session_state.eq is not None:
        st.write(expert_advice(query=st.session_state.eq)['experts'])
        st.write(expert_advice(query=st.session_state.eq)['answer'])

def main():
    st.set_page_config(page_title='Voice Assistant', page_icon="🎙️", layout='wide')

    # Create a navigation menu
    page = st.sidebar.radio('Navigation', ['Voice Assistant', 'Ask your PDF', 'DALL·E Generator','Plan Maker','Expert Advice'])

    if page == 'Voice Assistant':
        voice_assistant()
    elif page == 'Ask your PDF':
        ask_your_pdf()
    elif page == 'DALL·E Generator':
        dall_e_generator()
    elif page == 'Plan Maker':
        run_plan_maker()
    elif page == 'Expert Advice':
        expertAdvice()

if __name__ == '__main__':
    main()

