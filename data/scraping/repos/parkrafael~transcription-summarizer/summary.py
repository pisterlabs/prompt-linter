from openai import OpenAI
from pydub import AudioSegment

# api setup
import os
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

client = OpenAI(api_key = OPENAI_API_KEY)

# ====== HELPER: FILE DELETION ======
def delete_files_in_directory(directory_path):
    try:
        files = os.listdir(directory_path)

        for file in files:
            file_path = os.path.join(directory_path, file)

            if os.path.isfile(file_path):
                os.remove(file_path)
        print("All files deleted successfully")

    except OSError:
        print("Error occurred while deleting files")

# ====== TRANSCRIPTION ======
def transcription(file_name):
    try:
        file_path = 'audio/' + file_name
        audio = AudioSegment.from_mp3(file_path)

        # in milliseconds 
        chunk_length = 60 * 1000

        chunks = [audio[i:i + chunk_length] for i in range(0, len(audio), chunk_length)]

        # chunking audio clip to one minute chunks
        for i, chunk in enumerate(chunks):
            name_chunk = 'chunk_' + str(i) + '.mp3'
            name_chunk_path = 'chunks/' + name_chunk

            chunk.export(name_chunk_path, format="mp3")

        print(f"Successfully split the audio file into {len(chunks)} chunks")

    except OSError:
        print("Error occurred splitting audio file")

    # ====================================
    # transcribe each chunk 
    try: 
        transcription_array = []

        for i, chunk in enumerate(chunks):
            name_chunk = 'chunk_' + str(i) + '.mp3'
            name_chunk_path = 'chunks/' + name_chunk

            chunk = open(name_chunk_path, 'rb')

            transcription = client.audio.transcriptions.create(
                model = "whisper-1",
                file = chunk,
                response_format = 'text',
            )

            transcription_array.append(transcription)
            print(f"Successfully transcribed audio chunk {i}")

    except OSError:
        print("Error occurred transcribing audio chunks")

    # ====================================
    transcription_final = ''

    # append all strings in transcription_array to one string 
    for chunk in transcription_array:
        transcription_final = transcription_final + chunk

    delete_files_in_directory('chunks/')
    
    summary(transcription_final, file_name)

# ====== SUMMARY ======
def summary(transcription, file_name):
    try: 
        response = client.chat.completions.create(
            model="gpt-3.5-turbo-16k",
            temperature=0,
            messages=[
                {
                    "role": "system",
                    "content": "You are a proficient AI with a specialty in distilling information into key points. Based on the following text, identify and list the main points that were discussed or brought up. These should be the most important ideas, findings, or topics that are crucial to the essence of the discussion. Your goal is to provide a list that someone could read to quickly understand what was talked about. Do this in bullet form"
                },
                {
                    "role": "user",
                    "content": transcription
                }
            ]
        )

        tidy_file_name = file_name[:-4]
        tidy_file_name_path = 'summary/' + tidy_file_name + '_summarized' + '.md'

        # Access the content using the 'choices' attribute
        content = response.choices[0].message.content

        f = open(tidy_file_name_path, "w")
        f.write(content)
        f.close()

        print("Successfully summarized transcription")

    except OSError:
        print("Error occurred summarizing transcription")

# ====== TESTING ======
def main():
    print("Enter the name of your audio file (inclding file name at the end): ")
    transcription(input())
    print("All done! ๑(◕‿◕)๑ ")
    exit()

main()