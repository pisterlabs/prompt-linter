import cfg.cfg
from pathlib import Path
from openai import OpenAI
from datetime import datetime
from dateutil import rrule

client = OpenAI()


def tts_mp3_file(text):
    speech_file_path = Path(__file__).parent / "tts_speech1.mp3"
    response = client.audio.speech.create(
        model="tts-1",  # tts-1 tts-1-hd
        voice="echo",  # alloy, echo, fable, onyx, nova, and shimmer
        input=text
    )
    response.stream_to_file(speech_file_path)


if __name__ == "__main__":
    # text = input("Enter text: ")
    start_time = datetime.now()
    text = "你好呀，今天天气明媚，你想去哪里玩么？ Today is a wonderful day to build something people love!"
    tts_mp3_file(text)
    seconds = rrule.rrule(freq=rrule.SECONDLY, dtstart=start_time, until=datetime.now())
    print(f"total spend: {seconds.count()} seconds")
