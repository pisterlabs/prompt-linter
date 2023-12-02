import openai

if __name__ == "__main__":
    audio_file = open("/Users/huangsiyu/WORK/SteamAI/RID/src/data/jfk.flac", "rb")
    openai.api_key = "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"
    transcript = openai.Audio.transcribe("whisper-1", audio_file)
    print(transcript.text)