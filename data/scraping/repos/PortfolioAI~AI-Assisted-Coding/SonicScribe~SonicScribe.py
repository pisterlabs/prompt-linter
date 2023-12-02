import openai
import sounddevice as sd
import soundfile as sf
import os
from datetime import datetime

# OpenAI API Configuration
openai.api_key = "ENTER_YOUR_API_KEY"
openai.organization = "ENTER_OPENAI_ORGANIZATION"

def transcribe_audio(file_path):
    try:
        audio_file = open(file_path, "rb")
        transcript = openai.Audio.transcribe("whisper-1", audio_file)
        print(f'Transcript: {transcript}')
    except Exception as e:
        print(f"An error occurred while transcribing: {e}")

def record_audio(duration, output_dir):
    fs = 44100  # Sample rate
    seconds = duration  # Duration of recording

    try:
        print(f"Recording for {seconds} seconds...")
        myrecording = sd.rec(int(seconds * fs), samplerate=fs, channels=2)
        sd.wait()  # Wait until recording is finished

        filename = f'{output_dir}/{datetime.now().strftime("%Y%m%d%H%M%S")}.wav'
        sf.write(filename, myrecording, fs)
        print(f"Recording saved as {filename}")
        return filename
    except Exception as e:
        print(f"An error occurred while recording: {e}")
        return None

def main():
    recording_duration = 10  # default recording duration
    output_dir = ""C:\""  # default output directory

    while True:
        print("\n1. Set recording duration")
        print("2. Set output directory")
        print("3. Transcribe an audio file")
        print("4. Transcribe from microphone")
        print("5. Exit program")

        user_choice = input("Please enter your choice (1, 2, 3, 4 or 5): ")

        if not user_choice.isdigit() or int(user_choice) not in [1, 2, 3, 4, 5]:
            print("Invalid input. Please enter a number (1, 2, 3, 4 or 5).")
            continue

        user_choice = int(user_choice)

        if user_choice == 1:
            recording_duration = int(input("Enter the new recording duration (in seconds): "))
        elif user_choice == 2:
            output_dir = input("Enter the output directory path: ")
            if not os.path.isdir(output_dir):
                print(f"No such directory exists at {output_dir}")
                output_dir = "."
        elif user_choice == 3:
            file_path = input("Please enter the path to your audio file: ")
            if os.path.exists(file_path):
                transcribe_audio(file_path)
            else:
                print(f"No such file exists at {file_path}")
        elif user_choice == 4:
            filename = record_audio(recording_duration, output_dir)
            if filename is not None:
                transcribe_audio(filename)
        elif user_choice == 5:
            print("Exiting the program...")
            break

if __name__ == "__main__":
    main()
