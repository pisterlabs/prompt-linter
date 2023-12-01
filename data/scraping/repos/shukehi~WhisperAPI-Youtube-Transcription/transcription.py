import openai
from videoproc import youtube_preprocess
from chunking import chunk_by_size


openai.api_key="YOUR_KEY_HERE"

audio_file = youtube_preprocess("YOUR_LINK_HERE")

no_of_chunks = chunk_by_size(audio_file)

for i in range(no_of_chunks):
    print(f"process_chunks/chunk{i}.wav")
    curr_file = open(f"process_chunks/chunk{i}.wav", "rb")
    transcript = openai.Audio.translate("whisper-1", curr_file)

    with open("videotext.txt","a") as f:
        f.write(transcript["text"])