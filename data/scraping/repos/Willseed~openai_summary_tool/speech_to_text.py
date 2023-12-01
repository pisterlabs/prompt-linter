import openai
from tqdm import tqdm

class SpeechToText:

    def __init__(self, token: str, filename: str, file_amount: int) -> None:
        openai.api_key = token
        self.filename = filename
        self.file_amount = file_amount

    def output_text_result(self) -> None:
        for i in tqdm(range(1, self.file_amount + 1)):
            audio_file = open('output_%s.mp3' % i, 'rb')
            transcript = openai.Audio.transcribe("whisper-1", audio_file)
            text = ''.join(transcript.text.split())
            with open(self.filename, 'a', encoding='utf-8') as f:
                f.write(text + '\n')
