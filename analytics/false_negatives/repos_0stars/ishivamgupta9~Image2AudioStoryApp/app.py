import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from transformers import pipeline
from langchain import PromptTemplate, OpenAI, LLMChain
from dotenv import load_dotenv
import requests
import streamlit as st
from PIL import Image

load_dotenv()

HUGGINGFACE_API_TOKEN = os.getenv("HUGGINGFACE_API")

# Streamlit setup

st.title("Image to Story App")


def main():
    file = st.file_uploader("Choose an image..", type=["jpg", "jpeg", "png", "gif"])

    if file is not None:
        data = file.getvalue()
        with open(file.name, 'wb') as image_file:
            image_file.write(data)

        # Load and validate the image using Pillow (PIL)
        try:
            img = Image.open(file.name)
            if img.format in ["JPEG", "PNG", "JPG", "GIF"]:  # Add or remove formats as needed
                st.image(img, caption="Uploaded Image", use_column_width=True)
                scenario = imgtext(file.name)
                story = generatestory(scenario)
                text2speech(story)
                

                with st.expander("Scenario"):
                    st.write(scenario)

                with st.expander("Story"):
                    st.write(story)

                st.audio("audio.flac")
            else:
                st.write("Invalid image format. Please upload a valid image.")
        except Exception as e:
            st.write(f"Error while processing the image: {str(e)}")






def imgtext(image_name):
    try:
        captioner = pipeline("image-to-text", model="Salesforce/blip-image-captioning-base")
        res = captioner(image_name)[0]["generated_text"]
        print("Image Caption:", res)
        return res
    except Exception as e:
        print("An error occurred:", str(e))
        return None


def generatestory(scenario):
    template = """
    You are a storyteller, so generate a short story, no more than 20 words;
    CONTEXT: {scenario}
    STORY:
    """
    prompt = PromptTemplate(template=template, input_variables=["scenario"])
    story_llm = LLMChain(llm=OpenAI(model_name="gpt-3.5-turbo", temperature=1), prompt=prompt, verbose=True)
    story = story_llm.predict(scenario=scenario)

    print(story)

    return story


def text2speech(message):
    API_URL = "https://api-inference.huggingface.co/models/espnet/kan-bayashi_ljspeech_vits"
    headers = {"Authorization": f"Bearer {HUGGINGFACE_API_TOKEN}"}

    payload = {
        "inputs": message
    }

    response = requests.post(API_URL, headers=headers, json=payload)

    if response.status_code == 200:
        with open("audio.flac", 'wb') as audio_file:
            audio_file.write(response.content)
    else:
        st.write("Error:", response.status_code)
        st.write("Response content:", response.content)


# Run the Streamlit app
if __name__ == "__main__":
    main()
