import os
import whisper
import openai
from gtts import gTTS
from datetime import datetime
from config import API_KEY


class V2V:
    def __init__(self, file_path):
        self.file_path = file_path

    def process(self):
        model = whisper.load_model("base")
        openai.api_key = API_KEY

        # load audio and pad/trim it to fit 30 seconds
        audio = whisper.load_audio(self.file_path)
        audio = whisper.pad_or_trim(audio)

        # make log-Mel spectrogram and move to the same device as the model
        mel = whisper.log_mel_spectrogram(audio).to(model.device)

        # detect the spoken language
        _, probs = model.detect_language(mel)
        language = {max(probs, key=probs.get)}
        print(f"Detected language: {max(probs, key=probs.get)}")

        # decode the audio
        options = whisper.DecodingOptions()
        result = whisper.decode(model, mel, options)

        # print the recognized text
        print(result.text)

        response = openai.Completion.create(
            model="text-davinci-003",
            prompt=result.text,
            temperature=0.7,
            max_tokens=255,
        )

        output = response.choices[0].text

        recorder = gTTS(text=output)

        if not os.path.exists('./generated'):
            os.mkdir('./generated')

        file_name = datetime.now().second

        recorder.save(f"./generated/{file_name}-output.mp3")

        print(f"Saved as {file_name}-output.mp3")

        return f"./generated/{file_name}-output.mp3"
