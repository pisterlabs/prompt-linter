import cfg.cfg
import io
from openai import OpenAI
from pydub import AudioSegment
from pydub.playback import play
from datetime import datetime
from dateutil import rrule

client = OpenAI()

# pip install openai
# pip install pydub


def stream_and_play(text):
    start_time = datetime.now()
    response = client.audio.speech.create(
        model="tts-1-hd-1106",  # tts-1 tts-1-hd  tts-1-1106 tts-1-hd-1106
        voice="echo",  # alloy, echo, fable, onyx, nova, and shimmer
        input=text,
    )

    # Convert the binary response content to a byte stream
    byte_stream = io.BytesIO(response.content)

    # Read the audio data from the byte stream
    audio = AudioSegment.from_file(byte_stream, format="mp3")

    # Play the audio
    play(audio)

    seconds = rrule.rrule(freq=rrule.SECONDLY, dtstart=start_time, until=datetime.now())
    print(f"total spend: {seconds.count()} seconds")


if __name__ == "__main__":
    # text = input("Enter text: ")
    text = "亲，江浙沪包邮哦～ 你好呀，今天天气明媚，你想去哪里玩么?"
    stream_and_play(text)
