import os
import io
import time
import openai
import sounddevice as sd
import numpy as np
import requests
import pyaudio
import wave
from elevenlabs import clone, generate, play, set_api_key
import soundfile as sf
import threading

openai.api_key = '#OpenAI API'
set_api_key("ElevenLabs API Key")

silence_threshold = 100  # The volume level below which to consider audio as silence
batch_number = 1  # Add batch_number as a global variable
transcription_threads = []
transcription_results = []

def record_audio(timestamp):
    sample_rate = 16000

    # Check if the directory exists, and create it if necessary
    if not os.path.exists('Masters Voice'):
        os.makedirs('Masters Voice')

    # Initialize PyAudio
    audio = pyaudio.PyAudio()

    # Start a new recording session
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    print("Recording...")
    stream = audio.open(format=pyaudio.paInt16, channels=1,
                        rate=sample_rate, input=True,
                        frames_per_buffer=1024)  # Feel free to adjust buffer size

    frames = []
    silence_duration = 3  # The duration of silence in seconds to wait before stopping recording
    silence_sample_count = silence_duration * sample_rate / 1024  # The number of silence samples to wait before stopping recording

    silent_samples = 0
    batch_silent_samples = 0
    batch_silence_sample_count = sample_rate / 1024  # Number of samples for 1 second of silence

    # Record until enough silence has been detected
    while silent_samples < silence_sample_count:
        data = stream.read(1024)
        frames.append(data)

        # Calculate volume
        np_data = np.frombuffer(data, dtype=np.int16).astype(np.int64)
        volume = np.sqrt(np.mean(np_data ** 2))

        # If volume is below threshold, increment silence count
        if volume < silence_threshold:
            silent_samples += 1
            batch_silent_samples += 1
        else:
            silent_samples = 0  # Reset the silence count when speech is detected
            batch_silent_samples = 0  # Reset the batch silence count when speech is detected

        # If enough silence for a batch is detected, save the batch and start a new one
        if batch_silent_samples >= batch_silence_sample_count:
            save_batch(frames, audio, timestamp, sample_rate)
            frames = []
            batch_silent_samples = 0

    # Stop and close the stream
    stream.stop_stream()
    stream.close()

    # Terminate the session
    audio.terminate()

    print("Recording completed!")


def save_batch(frames, audio, timestamp, sample_rate):
    # Calculate volume
    np_frames = np.frombuffer(b''.join(frames), dtype=np.int16).astype(np.int64)
    volume = np.sqrt(np.mean(np_frames ** 2))
    # Only save the batch if volume is above the silence threshold
    if volume >= silence_threshold:
        # Save the recorded data to a .wav file
        global batch_number

        wf = wave.open(get_batch_filename(timestamp, batch_number), 'wb')
        wf.setnchannels(1)
        wf.setsampwidth(audio.get_sample_size(pyaudio.paInt16))
        wf.setframerate(sample_rate)
        wf.writeframes(b''.join(frames))
        wf.close()

        # Start a new thread to transcribe the batch
        thread = threading.Thread(target=transcribe_and_store_result,
                                  args=(get_batch_filename(timestamp, batch_number), batch_number))
        thread.start()
        transcription_threads.append(thread)

        batch_number += 1


def get_batch_filename(timestamp, batch_number):
    return f'Masters Voice/{timestamp}_Batch{batch_number}.wav'  # Without 'Batch' prefix


def process_batches(timestamp):
    batch_files = [f for f in os.listdir('Masters Voice') if f.startswith(timestamp)]
    batch_files.sort()

    print(f"Batch files: {batch_files}")

    for i, batch_file in enumerate(batch_files):
        print(f"Transcribing file: {batch_file}")

    # Sort the transcriptions by batch number and combine them
    transcription_results.sort(key=lambda x: x[0])  # Sort by batch number
    user_message = ' '.join(transcription[1] for transcription in transcription_results)  # Combine the transcriptions

    return user_message


def transcribe_and_store_result(audio_file_path, batch_number):
    transcription = transcribe_audio(audio_file_path)
    print(f"Batch files: ['{os.path.basename(audio_file_path)}']")
    print(f"Transcribing file: {os.path.basename(audio_file_path)}")
    print(f"Transcription: {transcription}")
    transcription_results.append((batch_number, transcription))


def transcribe_audio(audio_file_path):
    audio_file = open(audio_file_path, "rb")
    response = openai.Audio.transcribe("whisper-1", audio_file)
    return response['text']


def play_audio(file_path):
    data, samplerate = sf.read(file_path)
    sd.play(data, samplerate)
    sd.wait()


def main():
    os.makedirs('Masters Voice', exist_ok=True)
    os.makedirs('Vex Responses', exist_ok=True)
    duration = 10
    sample_rate = 16000

    conversation_history = [
        {"role": "system",
         "content": "You are assuming the role of Vex, (personality goes here.)"},
        {"role": "user", "content": "Hello Vex, Let's have a conversation."},
    ]

    with open('conversation.txt', 'a') as file:
        while True:
            timestamp = time.strftime("%Y%m%d-%H%M%S")

            # initialize conversation history inside the loop
            conversation_history = []

            record_audio(timestamp)

            # Wait for all transcription threads to finish
            for thread in transcription_threads:
                thread.join()

                print("Transcribing...")
                user_message = process_batches(timestamp)  # Process all batches in the current session

            # Clear the lists for the next conversation round
            transcription_results.clear()
            transcription_threads.clear()

            print(f"[Master]: {user_message if user_message else 'No transcription'}")
            confirmation = input("Press enter to send this message to Vex, or type a new message: ")

            if confirmation:
                user_message = confirmation

            # Read the entire conversation from the file
            with open('conversation.txt', 'r') as read_file:
                lines = read_file.readlines()

            # Parse the lines to append to the conversation_history
            for line in lines:
                try:
                    if line.startswith("[Master]:"):
                        # line contains "[Master]: ", it is safe to split
                        message = line.split("[Master]: ", 1)[1].strip()
                        conversation_history.append({"role": "user", "content": message})
                    elif line.startswith("[Vex]:"):
                        # line contains "[Vex]: ", it is safe to split
                        message = line.split("[Vex]: ", 1)[1].strip()
                        conversation_history.append({"role": "assistant", "content": message})
                except IndexError:
                    # Line format was not as expected, so skipping it
                    continue

            # Append Master's message and system message to the conversation_history
            conversation_history.append({"role": "user", "content": user_message})
            system_message = {"role": "system",
                              "content": "You are assuming the role of Vex, (personality goes here.)"}
            conversation_history.append(system_message)

            # Append the Master's message to the conversation.txt
            file.write("-----\n")
            file.write(f"[Master]: {user_message if user_message else 'No transcription'}\n")
            file.flush()

            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=conversation_history,
            )
            assistant_response = response['choices'][0]['message']['content']
            print(f"[Vex]: {assistant_response}")

            audio = generate(text=assistant_response, voice="Voice Code from ElevenLabs")
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            output_filename = f"Vex Responses/output_{timestamp}.wav"
            with open(output_filename, "wb") as f:
                f.write(audio)

            # Play the generated audio file
            play_audio(output_filename)

            conversation_history.append({"role": "assistant", "content": assistant_response})
            file.write("-----\n")
            file.write(f"[Vex]: {assistant_response}\n")
            file.flush()

            if conversation_history[-1]['content'].strip().lower() == "Pineapple Juice":
                print("[Vex]: That's the magic word! Take care!!")
                file.write("-----\n")
                file.write("[Vex]: That's the magic word! Take care!!\n")
                file.flush()
                for i in range(3, 0, -1):
                    print(f"Closing in {i}...")
                    time.sleep(1)
                break


if __name__ == "__main__":
    main()
