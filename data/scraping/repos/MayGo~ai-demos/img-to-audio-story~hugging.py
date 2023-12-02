from dotenv import load_dotenv, find_dotenv
from transformers import pipeline
from langchain import PromptTemplate, LLMChain, OpenAI
from langchain.chat_models import ChatOpenAI
import requests
import os
import streamlit as st

from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
from datasets import load_dataset
import torch
import soundfile as sf
from datasets import load_dataset

load_dotenv(find_dotenv())


tempFolder = "temp"

HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_TOKEN")

if not os.path.exists(tempFolder):
    os.makedirs(tempFolder)


def img2text(url):
    image_to_text = pipeline(
        "image-to-text", model="Salesforce/blip-image-captioning-large"
    )

    text = image_to_text(url, max_new_tokens=500)[0]["generated_text"]
    print(text)
    return text


def generate_story(scenario):
    template = """
    You are a story teller.
    You can a short based on a simple scenario, the story should be no more than 50 words

    CONTEXT: {scenario}
    STORY:
    """

    prompt = PromptTemplate(template=template, input_variables=["scenario"])
    story_llm = LLMChain(
        llm=ChatOpenAI(model_name="gpt-3.5-turbo", temperature=1),
        prompt=prompt,
        verbose=True,
    )

    story = story_llm.predict(scenario=scenario)

    print(story)
    return story


def text2speech(message):
    API_URL = (
        "https://api-inference.huggingface.co/models/espnet/kan-bayashi_ljspeech_vits"
    )
    headers = {"Authorization": f"Bearer {HUGGINGFACE_TOKEN}"}

    payloads = {"inputs": message}

    response = requests.post(API_URL, headers=headers, json=payloads)

    with open("audio.flac", "wb") as file:
        file.write(response.content)


def text2advanced_speech(message):
    processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
    model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts")
    vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")

    inputs = processor(text=message, return_tensors="pt")

    # load xvector containing speaker's voice characteristics from a dataset
    embeddings_dataset = load_dataset(
        "Matthijs/cmu-arctic-xvectors", split="validation"
    )
    speaker_embeddings = torch.tensor(embeddings_dataset[7306]["xvector"]).unsqueeze(0)

    speech = model.generate_speech(
        inputs["input_ids"], speaker_embeddings, vocoder=vocoder
    )

    sf.write(os.path.join(tempFolder, "speech.wav"), speech.numpy(), samplerate=16000)


def main():
    st.set_page_config(page_title="img 2 audio story", page_icon="🐶", layout="centered")
    st.header("Turn image into story")
    uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg"])

    if uploaded_file is not None:
        bytes_data = uploaded_file.getvalue()

        # open with a path folder and write the file
        with open(os.path.join(tempFolder, uploaded_file.name), "wb") as file:
            file.write(bytes_data)

        st.image(uploaded_file, caption="Uploaded Image.", use_column_width=True)
        scenario = img2text(os.path.join(tempFolder, uploaded_file.name))
        story = generate_story(scenario)
        text2advanced_speech(story)

        with st.expander("scenario"):
            st.write(scenario)
        with st.expander("story"):
            st.write(story)
        st.audio(os.path.join(tempFolder, "speech.wav"))


if __name__ == "__main__":
    main()
