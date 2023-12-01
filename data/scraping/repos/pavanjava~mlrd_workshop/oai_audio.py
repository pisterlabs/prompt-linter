import os
from openai import OpenAI
from dotenv import load_dotenv, find_dotenv

_ = load_dotenv(find_dotenv())  # read local .env file


class OpenAIAudioAgent:
    def __init__(self):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    def transcribe(self, file):
        with open(file=file, mode='rb') as audio_file:
            response = self.client.audio.transcriptions.create(
                model='whisper-1',
                file=audio_file
            )

            return response.text

    def translate(self, transcribed_text, target_language):
        return self.client.completions.create(
            model="gpt-3.5-turbo-instruct",
            prompt=f"as professional translating assistant, your job is to translate the given text in triple back tick to {target_language} context: ```{transcribed_text}```",
            temperature=0.5,
            max_tokens=500,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0
        )

    def to_audio(self, translated_text):
        speech_file_path = "translated_speech.mp3"
        resp = self.client.audio.speech.create(
            model="tts-1",
            voice="alloy",
            input=str(translated_text)
        )
        resp.stream_to_file(speech_file_path)


if __name__ == "__main__":
    obj = OpenAIAudioAgent()
    transcript = obj.transcribe(file="/Users/pavanmantha/Pavans/Workshops/Mallareddy-university/tutorials/section-1/hear-and-translate/sample.m4a")
    print(transcript)
    target_text = obj.translate(transcribed_text=transcript, target_language="hindi")
    print(target_text.choices[0].text)
    obj.to_audio(translated_text=target_text.choices[0].text)






