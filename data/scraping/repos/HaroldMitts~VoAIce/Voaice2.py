import json
import os
import subprocess
import tempfile
import time
import azure.cognitiveservices.speech as speechsdk
import openai

def load_api_keys(file_path):
    with open(file_path, "r") as f:
        keys = json.load(f)
    return keys

def transcribe_audio(speech_config):
    audio_config = speechsdk.AudioConfig(use_default_microphone=True)
    speech_recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config, audio_config=audio_config)

    result = speech_recognizer.recognize_once_async().get()
    return result.text.strip()

def generate_response(input_text, conversation_history):

    messages = [
        {"role": "system", "content": "Think step by step before answering any question. \
            reflect on the question and all potential answers. Examine the flaw and faulty \
            logic of each answer option and eliminate the ones that are incorrect. \
            If you are unsure of the answer, make an educated guess. \
            If you are still unsure, eliminate the answers that you know are incorrect and then guess \
            from the remaining answers. \
            If any answer provided is comprised of an educated guess, make sure to note that in your answer."},
    ]
    messages.extend(conversation_history)
    messages.append({"role": "user", "content": input_text})

    response = openai.ChatCompletion.create(
        #model="gpt-3.5-turbo",
        model="gpt-4",
        messages=messages,
        max_tokens=200,
        n=1,
        stop=None,
        temperature=1.3,
    )

    assistant_response = response['choices'][0]['message']['content']
    return assistant_response

def synthesize_and_save_speech(speech_config, response_text, file_path):
    speech_synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config)
    result = speech_synthesizer.speak_text_async(response_text).get()

    with open(file_path, "wb") as f:
        f.write(result.audio_data)

def play_audio(audio_file_path):
    subprocess.call(["ffplay", "-nodisp", "-autoexit", audio_file_path], stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)

def remove_temp_files(file_path):
    os.remove(file_path)

def main(quit_phrases=["I quit", "quit", "goodbye", "stop", "exit"]): #exit keywords
    keys_file_path = r"C:\\keys\\keys.json"
    keys = load_api_keys(keys_file_path)
    #keys = load_api_keys("/path/in/container/keys.json") #save your keys and Azure region to C:\GitHub\VoAI\VoAI\keys.json or change this path as needed.
    azure_api_key = keys["azure_api_key"]
    azure_region = keys["azure_region"]
    openai_api_key = keys["openai_api_key"]
    voice = "en-US-SaraNeural" # Set the voice here
    speech_config = speechsdk.SpeechConfig(subscription=azure_api_key, region=azure_region)
    speech_config.speech_synthesis_voice_name = voice  
    openai.api_key = openai_api_key

    conversation_history = []

    while True:
        print("Listening...")

        input_text = transcribe_audio(speech_config)
        print(f"Input: {input_text}")

        if any(phrase.lower() in input_text.lower() for phrase in quit_phrases):
            break

        response_text = generate_response(input_text, conversation_history)
        print(f"Response: {response_text}")

        conversation_history.append({"role": "user", "content": input_text})
        conversation_history.append({"role": "assistant", "content": response_text})

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            audio_file_path = f.name

        try:
            synthesize_and_save_speech(speech_config, response_text, audio_file_path)
        except Exception as e:
            print(f"Error: Failed to synthesize speech - {e}")

        # Play the WAV file
        try:
            print("Audio playback complete.")
        except Exception as e:
            print(f"Error: Failed to play WAV - {e}")

        # Remove the temporary files
        try:
            remove_temp_files(audio_file_path)
        except Exception as e:
            print(f"Error: Failed to remove temporary files - {e}")

        #time.sleep(1)  # Add a 1-second delay before the loop starts listening again

if __name__ == "__main__":
    main(quit_phrases=["I quit", "stop", "exit"])
