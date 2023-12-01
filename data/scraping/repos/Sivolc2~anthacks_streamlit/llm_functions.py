# llm_functions.py
import base64
import os
from pathlib import Path

import numpy as np
import requests

# Import necessary libraries
import streamlit as st
import whisper
from elevenlabs import generate as generate_audio
from elevenlabs import set_api_key as elevenlabs_set_api_key
from langchain.agents import Tool, create_csv_agent, initialize_agent, load_tools
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chat_models import ChatAnthropic
from langchain.llms import OpenAI
from langchain.memory import ConversationBufferMemory
from langchain.prompts.chat import (
    AIMessagePromptTemplate,
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)
from langchain.schema import AIMessage, HumanMessage, SystemMessage
from langchain.utilities import PythonREPL

from audio_functions import autoplay_audio

## Get Key
elevenlabs_set_api_key(os.getenv("ELEVENLABS_API_KEY"))


def translation_page():
    # Ask the user for a sentence to translate
    user_sentence = st.text_input(
        "Please enter a sentence to translate from English to French:"
    )

    # Only attempt to translate when the user has entered a sentence
    if user_sentence:
        chat = ChatAnthropic()

        messages = [
            HumanMessage(
                content=f"Translate this sentence from English to French. {user_sentence}"
            )
        ]
        response = chat(messages)

        # Display the translation
        st.write(f"The translated sentence is: {response.content}")


def chat_page():
    import anthropic

    anthropic_client = anthropic.Client(api_key=os.getenv("ANTHROPIC_API_KEY"))

    context = ""
    user_inp = st.text_input("You: ")

    if user_inp:
        current_inp = anthropic.HUMAN_PROMPT + user_inp + anthropic.AI_PROMPT
        context += current_inp

        prompt = context

        completion = anthropic_client.completion(
            prompt=prompt, model="claude-v1.3-100k", max_tokens_to_sample=1000
        )["completion"]

        context += completion

        # Display the response from the model
        st.write("Anthropic: " + completion)

        # Generate an audio file with the response and play it
        audio = generate_audio(
            text=completion, voice="Arnold", model="eleven_multilingual_v1"
        )
        audio_base64 = base64.b64encode(audio).decode("utf-8")
        audio_tag = f'<audio autoplay src="data:audio/ogg;base64,{audio_base64}">'
        st.markdown(audio_tag, unsafe_allow_html=True)


def repl_agent_page():
    import anthropic
    st.title("REPL Agent Page")

    # Create a file uploader for the user to upload a pdf
    pdf_file = st.file_uploader("Please upload a PDF file", type=["pdf"])

    # Prompt the user for a question about the pdf
    user_question = st.text_input("Enter your question about the PDF:")

    # If a pdf file has been uploaded and a question has been asked, process it
    if pdf_file is not None and user_question:
        # Save the pdf file to a temporary location
        pdf_path = "temp_pdf_file.pdf"
        with open(pdf_path, "wb") as f:
            f.write(pdf_file.getbuffer())
            pdf_scan = f.read()
        

        python_repl = PythonREPL()
        repl_tool = Tool(
            name="python_repl",
            description="A Python shell. Use this to execute python commands. Input should be a valid python command. If you are done with your observations you should return the answer in plain text",
            func=python_repl.run,
        )

        tools = [repl_tool]
        memory = ConversationBufferMemory(memory_key="chat_history")
        llm = ChatAnthropic()
        
        if user_sentence:
            chat = ChatAnthropic()

            messages = [
                HumanMessage(
                    content=f"Summarize this PDF {pdf_scan}"
                )
            ]
            response = chat(messages)

            # Display the translation
            st.write(f"A summary of this pdf is {response.content}")


        # conversational_agent = initialize_agent(
        #     agent="conversational-react-description",
        #     tools=tools,
        #     llm=llm,
        #     verbose=True,
        #     max_iterations=3,
        #     memory=memory,
        # )

        # # Pass the pdf path and user's question to your agent for processing
        # agent = conversational_agent(
        #     f"""When responding please, please output a response in this format:
        #     thought: Reason about what action to take next, and whether to use a tool.
        #     action: The tool to use. Must be one of: ({repl_tool})
        #     action: The tool to use. Either no tool or python_repl
        #     action_input: The input to the tool
        #     For example:
        #     thought: I need to send a message to xxx
        #     action: Telegram
        #     action_input: Send a message to xxx: I don't agree with that idea.
        #     Now, read the file `{pdf_path}`. {user_question}
        # """
        # )
        
        
        # Delete the temporary pdf file after processing
        os.remove(pdf_path)
        
        st.write(agent)

    else:
        st.write("Please upload a PDF file and enter your question for processing.")


class MediaManager:
    """A class to act as a primary interface to manage media objects and related data"""

    def __init__(self, media_dir: str = "MEDIA_DIR"):
        self.media_dir = Path(media_dir)
        self.media_dir.mkdir(exist_ok=True, parents=True)

    def _transcribe(self, audio_path: str, whisper_model: str):
        """Transcribe the audio file using whisper"""

        # Check if the file has content
        if os.path.getsize(audio_path) == 0:
            st.write("Error: The audio file has no content.")
            return None

        # Load whisper model
        model = whisper.load_model(whisper_model)

        try:
            # Transcribe the audio file
            transcript = model.transcribe(audio_path)
        except Exception as e:
            st.write(f"Error during transcription: {str(e)}")
            st.write(f"Audio file path: {audio_path}")
            return None

        return transcript['text']  # return only the transcribed text, not the entire output
