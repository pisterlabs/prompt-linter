import streamlit as st
import time
from google.auth import credentials
from google.oauth2 import service_account
import google.cloud.aiplatform as aiplatform
from vertexai.preview.language_models import ChatModel, TextGenerationModel
import vertexai
import json
import openai
import os
from dotenv import load_dotenv
from PIL import Image
from gtts import gTTS
from io import BytesIO
import base64



# Load the service account json file
# Update the values in the json file with your own
with open(
    "service_account.json"
) as f:  # replace 'serviceAccount.json' with the path to your file if necessary
    service_account_info = json.load(f)

my_credentials = service_account.Credentials.from_service_account_info(
    service_account_info
)

# Initialize Google AI Platform with project details and credentials
aiplatform.init(
    credentials=my_credentials,
)

with open("service_account.json", encoding="utf-8") as f:
    project_json = json.load(f)
    project_id = project_json["project_id"]


# Initialize Vertex AI with project and location
vertexai.init(project=project_id, location="us-central1")

load_dotenv()


SECRET_KEY = os.getenv("OPEN_AI_KEY")
openai.api_key = SECRET_KEY

st.title(":green[Gyan Chat]")

## Store LLM generated responses
if "messages" not in st.session_state.keys():
    st.session_state.messages = [{"role": "assistant", "content": "How may I help you?"}]

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Function for generating LLM response
def generate_response(user_input):
    # Create ChatBot                        
    chatbot = ChatModel.from_pretrained("chat-bison@001")
    parameters = {
        "temperature": 0.2,
        "max_output_tokens": 256,
        "top_p": 0.8,
        "top_k": 40
    }
    chat = chatbot.start_chat(
        context="""Your name is GyanAI. You are an assistant who helps kids (age group 5-10 years) develop their communication skills.
    Praise the user (in creative ways) for asking questions to encourage them.
    Respond in \"short and simple sentences\". Shape your response as if talking to a 7-years-old.""",

    )
    response = chat.send_message(user_input, **parameters)
    output = response.text
    return output


def generate_prompt(user_input, response):
    model = TextGenerationModel.from_pretrained("text-bison@001")
    parameters = {
    "temperature": 0.2,
    "max_output_tokens": 256,
    "top_p": 0.8,
    "top_k": 40
    }
    prompt = model.predict(
    f"""Based on the Examples provided, extract the central theme of the most recent conversation, based on which generate a prompt giving vivid descriptions for Dall-E so that it can generate an image related to the topic for a 7 year old to help visualize and learn.

    input: what is space?
    response: Space is all around us. It's the place where everything happens, like the planets, stars, and galaxies.
    output_prompt: A picture of a dark blue sky with stars and planets. The stars are different colors and sizes. The planets are different shapes and sizes.

    input: what is a house?
    response: That's a great question! A house is a building where people live. It has rooms for sleeping, eating, and playing.
    output_prompt: A picture of a house with a red roof and white walls. The house has a front door and two windows.

    input: {user_input}
    response: {response}
    output_prompt: 
    """,
        **parameters
    )
    output = prompt.text
    return output


# User-provided prompt
if prompt := st.chat_input():
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

# Generate a new response if last message is not from assistant
if st.session_state.messages[-1]["role"] != "assistant":
    #with st.chat_message("assistant"):
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            message_placeholder = st.empty()
            full_response = ""
            response = generate_response(prompt) 
            for chunk in response.split():
                full_response += chunk + " "
                time.sleep(0.05)
                # Add a blinking cursor to simulate typing
                message_placeholder.markdown(full_response + "▌")
            message_placeholder.markdown(full_response)

            answer = full_response
            tts = gTTS(answer, lang='en', tld='co.in')
            tts.save('answer.wav')
            audio_byte = BytesIO()
            tts.write_to_fp(audio_byte)
            audio_byte.seek(0)

            st.audio(audio_byte, format="audio/wav")
            dalle_prompt = generate_prompt(prompt, response)
            image_gen = openai.Image.create(
                prompt=dalle_prompt,
                n=1,
                size="256x256",
            )
            image_url = image_gen['data'][0]['url']
            st.image(image_url)
    

    message = {"role": "assistant", "content": response}
    st.session_state.messages.append(message)
