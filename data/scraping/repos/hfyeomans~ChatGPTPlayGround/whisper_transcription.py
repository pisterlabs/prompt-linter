import pyaudio
import wave
import threading
import openai
import os

# Need to install the openai module before using
openai.api_key = os.environ['OPEN_AI_KEY']

# Parameters
output_file = "output.wav"
stop_recording = False
def record_audio(output_file):
    audio_format = pyaudio.paInt16
    channels = 1
    sample_rate = 44100
    chunk_size = 1024
# record_seconds = 5

# Create a PyAudio object
    p = pyaudio.PyAudio()

# Open a streaming stream
    stream = p.open(format=audio_format,
                    channels=channels,
                    rate=sample_rate,
                    input=True,
                    frames_per_buffer=chunk_size)

    print("Recording...")

    frames = []

    while not stop_recording:
        data = stream.read(chunk_size)
        frames.append(data)

    # Stop and close the stream
    print("Recording finished.")
    stream.stop_stream()
    stream.close()
    p.terminate()

    # Save the recorded data to a WAV file
    wf = wave.open(output_file, 'wb')   
    wf.setnchannels(channels)
    wf.setsampwidth(p.get_sample_size(audio_format))
    wf.setframerate(sample_rate)
    wf.writeframes(b''.join(frames))
    wf.close()

def wait_for_enter():
    global stop_recording
    input()
    stop_recording = True

def transcribe_audio(file_path):
    audio_file = open(file_path, "rb")
    try:
        response = openai.Audio.translate("whisper-1", audio_file)
    except openai.error.APIError as e:
        #Handle API error here, e.g. retry or log
        print(f"OpenAI API returned an API Error: {e}")
        pass
    except openai.error.APIConnectionError as e:
    #Handle connection error here
        print(f"Failed to connect to OpenAI API: {e}")
        pass
    except openai.error.RateLimitError as e:
    #Handle rate limit error (we recommend using exponential backoff)
        print(f"OpenAI API request exceeded rate limit: {e}")
        pass
    return response['text']


def get_whisper_response(prompt):
    transcription = transcribe_audio(output_file)
    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "assistant", "content": transcription}
        ]
        
    )
    return completion.choices[0].message['content']

        

def main():
    print("Press Enter to start recording, then press Enter again when you are finished recording.")
    input()
    recorder_thread = threading.Thread(target=record_audio, args=(output_file,))
    enter_thread = threading.Thread(target=wait_for_enter)

    recorder_thread.start()
    enter_thread.start()

    recorder_thread.join()
    enter_thread.join()

    print("Transcribing audio")
    transcribed_text = transcribe_audio(output_file)

    if transcribed_text:
        print(f"\n\nTranscription: {transcribed_text}")
    
    chatgpt_response = get_whisper_response(transcribe_audio)
    if chatgpt_response:
        print(f"\n\nChatGPT Response: \n{chatgpt_response}")

if __name__ == '__main__':
    main()
