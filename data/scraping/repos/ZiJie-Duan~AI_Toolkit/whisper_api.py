import openai

class AudioTranscriber:
    def __init__(self, api_key, model="whisper-1"):
        self.api_key = api_key
        self.model = model
        openai.api_key = self.api_key

    def transcribe_audio(self, file_path):
        with open(file_path, "rb") as audio_file:
            transcript = openai.Audio.transcribe(self.model, audio_file)
        return transcript

    def get_transcription(self, file_path):
        result = self.transcribe_audio(file_path)
        return result["text"]

# def main():
#     api_key = ""
#     file_path = "audio.mp3"

#     transcriber = AudioTranscriber(api_key)
#     transcription = transcriber.get_transcription(file_path)
#     print("Transcription result:", transcription)

# if __name__ == "__main__":
#     main()
