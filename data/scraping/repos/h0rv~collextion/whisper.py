from openai import Audio
from os import path
from pydub import AudioSegment


def call_whisper_api(mp3_path: str) -> str:
    """
    Whisper Docs: https://platform.openai.com/docs/guides/speech-to-text
    """
    with open(mp3_path, "rb") as audio_file:
        # TODO: Potentially prompt it with podcast description
        transcript = Audio.transcribe("whisper-1", audio_file)

    return transcript["text"]


def split_podcast_mp3(mp3_path: str):
    podcast_audio = AudioSegment.from_mp3(mp3_path)

    fname = mp3_path[:mp3_path.rfind(".mp3")]  # remove file ext

    # PyDub handles time in milliseconds
    ten_minutes = 10 * 60 * 1000

    counter = 1
    while podcast_audio is not []:
        first_10_minutes = podcast_audio[:ten_minutes]

        new_path = path.join(fname, "_", counter, ".mp3")

        first_10_minutes.export(new_path, format="mp3")

        counter += 1
        podcast_audio = podcast_audio[ten_minutes:]


def main():
    return


if __name__ == "__main__":
    main()
