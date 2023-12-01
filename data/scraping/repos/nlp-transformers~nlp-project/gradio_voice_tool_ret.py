#from transformers import pipeline
import whisper
import gradio as gr
from langchain.llms import OpenAI
from langchain.agents import load_tools, initialize_agent, AgentType
from langchain.tools import BaseTool, StructuredTool, Tool, tool
from deep_translator import GoogleTranslator
from random_number_tool import random_number_tool
from youTube_helper import youtube_tool
from url_scraping_tool import url_scraping_tool
from current_time_tool import current_time_tool
from wiki_tool import wiki_tool
from weather_tool import weather_tool
from langchain.agents import create_sql_agent
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from langchain.sql_database import SQLDatabase
from tool_retrieval import get_tools
from text_to_image import text_to_image
from text_to_video import text_to_video


from random_number_tool import random_number_tool
from youTube_helper import youtube_search

# ******** MAC-OS *************
# from AppKit import NSSpeechSynthesizer
# nssp = NSSpeechSynthesizer
# ve = nssp.alloc().init()

# **** GOOGLE TextToSpeech *****
from gtts import gTTS
import os

# Language in which you want to convert
language = 'en'

# to get input from speech use the following libs
model = whisper.load_model("large")

# define llm
llm = OpenAI(temperature=0.1)

postdb = SQLDatabase.from_uri("postgresql://abhi:mango@localhost:5432/abhi?sslmode=disable")
toolkit = SQLDatabaseToolkit(db=postdb, llm=llm)
sql_agent = create_sql_agent(
    llm=OpenAI(temperature=1.0),
    toolkit=toolkit,
    verbose=True
)

# define agent


# core function which will do all the work (POC level code)
def transcribe(audio, state=""):
    # load audio and pad/trim it to fit 30 seconds
    audio = whisper.load_audio(audio)
    audio = whisper.pad_or_trim(audio)

    # make log-Mel spectrogram and move to the same device as the model
    mel = whisper.log_mel_spectrogram(audio).to(model.device)
    # detect the spoken language
    _, probs = model.detect_language(mel)
    
    detected_language = max(probs, key=probs.get)
    print("detected_language --> ", detected_language)
    if detected_language == "en":
        print("Detected English language.")
        options = whisper.DecodingOptions(fp16=False, task="transcribe", language=language)
        result = whisper.decode(model, mel, options)
        result_text = result.text
    elif detected_language == "ta":
        print("Detected Tamil language.")
        options = whisper.DecodingOptions(fp16=False, task="transcribe", language="ta")
        tamil = whisper.decode(model, mel, options)
        print(tamil.text)
        result_text = GoogleTranslator(source='ta', target=language).translate(tamil.text)

        # transcribe = pipeline(task="automatic-speech-recognition", model="vasista22/whisper-tamil-medium",
        #                       chunk_length_s=30, device="cpu")
        # transcribe.model.config.forced_decoder_ids = transcribe.tokenizer.get_decoder_prompt_ids(language="ta", task="transcribe")

    else:
        result_text = "Unknown language"

    print("result text --> ", result_text)
    if result_text != "Unknown language" and len(result_text)!= 0:
        # Now add the lanfChain logic here to process the text and get the responses.
        # once we get the response, we can output it to the voice.
        #agent.
        tools = get_tools(result_text)
        agent = initialize_agent(tools=tools,  llm=llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)
        print("agent tool --> ", agent.tools)
        agent_output = agent.run(result_text)
    else:
        agent_output = "I'm sorry I cannot understand the language you are speaking. Please speak in English or Tamil."

    # init some default image and video. Override based on agent output.
    detailed = ''
    image_path = 'supportvectors.png'
    video_path = 'welcome.mp4'

    if "tool" in agent_output:
        print("This is an article.")
        tldr = agent_output["tldr"]
        detailed = agent_output["article"]
        if (agent_output["tool"] == "youtube") and ("video" in agent_output):
            video_path = agent_output["video"]

    else:
        print("This is not an article. It is coming from agent.", agent_output)
        tldr = agent_output


    # generate image based on tldr
    try:
        image = text_to_image(tldr)
        if image:
            image_path = 'output.png'
    except:
        print('Some problem generating image.')

    # generate image based on tldr
    try:
        tldr_video_path = text_to_video(tldr)
    except:
        print('Some problem generating video.')

    # TTS. Marked slow=False meaning audio should have high speed
    myobj = gTTS(text=tldr, lang=language, slow=False)
    # Saving the converted audio in a mp3 file named
    myobj.save("welcome.mp3")
    # Playing the audio
    # os.system("mpg123 welcome.mp3")

    return tldr, detailed, image_path, video_path, tldr, "welcome.mp3", tldr_video_path


# Set the starting state to an empty string
gr.Interface(
    fn=transcribe,
    inputs=[gr.Audio(source="microphone", type="filepath", streaming=False), "state"],
    outputs=["textbox", "textbox", "image", "video", "state", "audio", "video"],
    live=True,
).launch(share=True)
