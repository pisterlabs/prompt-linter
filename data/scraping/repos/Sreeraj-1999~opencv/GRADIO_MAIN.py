##################### GRADIO CODE ###################################

import gradio as gr
import numpy as np
import cv2
import pytesseract
import openai
import tempfile
from gtts import gTTS
import soundfile as sf
from gradio import components


pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'


openai.api_key = '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'


def text_extraction_app(image):

    img_array = np.array(image)


    img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

    text = pytesseract.image_to_string(img_bgr)
    text = text.replace("\n", "")

    prompt = f"Please provide a detailed explanation of the meaning of the following quote: '{text}'"
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=prompt,
        max_tokens=1000,
        temperature=0.1,
    )
    generated_translation = response.choices[0].text.strip()

    tts = gTTS(text=generated_translation, lang="en", slow=False)


    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_audio_file:
        temp_audio_path = temp_audio_file.name
        tts.save(temp_audio_path)


    data, sample_rate = sf.read(temp_audio_path)


    return generated_translation, temp_audio_path

iface = gr.Interface(
    fn=text_extraction_app,
    inputs=components.Image(type="pil"),
    outputs=[components.Textbox(label="GENERATED TEXT"),
             components.Audio(label="GENERATED AUDIO")]
)

if __name__ == "__main__":
    iface.launch()
