import sys
import os
import pyaudio
import queue
from google.cloud import speech_v1p1beta1 as speech
from google.cloud import texttospeech_v1 as tts
import openai
import json
import requests


os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "gcp_api.json"
with open('openai_api.json', 'r') as file:
    api_keys = json.load(file)
    openai.api_key = api_keys.get("OPENAI_API_KEY")

RATE = 16000
CHUNK = int(RATE / 10)  # 100ms
tts_client = tts.TextToSpeechClient()


class MicrophoneStream:
    def __init__(self, rate, chunk):
        self.rate = rate
        self.chunk = chunk
        self.buffers = queue.Queue()
        self.closed = True

    def __enter__(self):
        self.audio_interface = pyaudio.PyAudio()
        self.audio_stream = self.audio_interface.open(
            format=pyaudio.paInt16,
            channels=1, rate=self.rate,
            input=True, frames_per_buffer=self.chunk,
            stream_callback=self._fill_buffer,
        )
        self.closed = False
        return self

    def __exit__(self, type, value, traceback):
        self.audio_stream.stop_stream()
        self.audio_stream.close()
        self.closed = True
        self.buffers.put(None)
        self.audio_interface.terminate()

    def _fill_buffer(self, in_data, frame_count, time_info, status_flags):
        self.buffers.put(in_data)
        return None, pyaudio.paContinue

    def generator(self):
        while not self.closed:
            chunk = self.buffers.get()
            if chunk is None:
                return
            data = [chunk]
            while True:
                try:
                    chunk = self.buffers.get(block=False)
                    if chunk is None:
                        return
                    data.append(chunk)
                except queue.Empty:
                    break
            yield b"".join(data)


last_spoken = ""


def synthesize_speech(text):
    synthesis_input = tts.SynthesisInput(text=text)

    voice = tts.VoiceSelectionParams(
        language_code="en-US",
        name="en-US-Wavenet-C",
        ssml_gender=tts.SsmlVoiceGender.FEMALE
    )

    audio_config = tts.AudioConfig(
        audio_encoding=tts.AudioEncoding.LINEAR16,
        speaking_rate=1.3
    )

    response = tts_client.synthesize_speech(
        input=synthesis_input, voice=voice, audio_config=audio_config
    )

    play_audio(response.audio_content)


def play_audio(data):
    p = pyaudio.PyAudio()
    stream = p.open(
        format=p.get_format_from_width(2),
        channels=1,
        rate=24000,
        output=True,
    )
    stream.write(data)
    stream.stop_stream()
    stream.close()
    p.terminate()


def listen_print_loop(responses):
    for response in responses:
        if not response.results:
            continue

        result = response.results[0]
        if not result.alternatives:
            continue

        transcript = result.alternatives[0].transcript

        if result.is_final:
            print("User: ", transcript)
            return transcript


class SummaryBuilder:
    def __init__(self):
        self.summary = []

    def add_message(self, role, message):
        self.summary.append({"role": role, "message": message})

    def generate_summary(self):
        summary_text = ""
        for i, entry in enumerate(self.summary):
            summary_text += f"{i+1}. {entry['role'].title()}: {entry['message']}\n"
        return summary_text


class SummaryBuilder:
    def __init__(self):
        self.summary = []

    def add_message(self, role, message):
        self.summary.append({"role": role, "message": message})

    def generate_summary(self):
        summary_text = ""
        for i, entry in enumerate(self.summary):
            summary_text += f"{i+1}. {entry['role'].title()}: {entry['message']}\n"
        return summary_text


def summarize_with_openai(summary_text):
    messages = [
        {"role": "system", "content": "You are a summarizer. Provide a very short summary of the following conversation that will be used to brief a 911 operator."},
        {"role": "user", "content": summary_text}
    ]

    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=messages,
    )
    summarized_text = response["choices"][0]["message"]["content"]
    return summarized_text.strip()


summary_builder = SummaryBuilder()


def run_conversation(temp):

    url = f"https://2478-2610-148-1f02-3000-9d36-aae4-265d-97d.ngrok-free.app/calls/{temp}"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
    else:
        print(f"Request failed with status code {response.status_code}")

    location = data["location"]
    # medical emergency, fire, criminal activity, traffic accident, public safety threat
    nature_of_emergency = data["emergencyType"]
    your_name = data["name"]

    messages = [
        {"role": "system", "content": "You are a 911 operator and have a critical responsibility. You've just received an emergency call. The caller's name is '{}', and they have reported an emergency situation at location '{}', specifically related to '{}'. Your duty is vital: you must remain calm, assertive, and supportive to assist the caller effectively. It is imperative to extract precise and actionable information about the emergency, focusing on details like physical descriptions, specific behaviors, and distinctive characteristics of the incident or involved parties. Your questions should be formulated one at a time, be concise, clear, straightforward, and directly aimed at obtaining detailed information to assist emergency services in responding effectively and efficiently. Ensure your interactions provide reassurance to the caller while maintaining urgency to gather requisite information. Keep your response limited to 15 words."
         .format(
             your_name, location, nature_of_emergency)},
        {"role": "assistant",
            "content": "911, what's your emergency?".format(location)},
    ]

    synthesize_speech(messages[-1]['content'])

    language_code = "en-US"
    client = speech.SpeechClient()
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=RATE,
        language_code=language_code,
    )
    streaming_config = speech.StreamingRecognitionConfig(
        config=config, interim_results=True)

    for i in range(4):
        print("\nListening...")

        with MicrophoneStream(RATE, CHUNK) as stream:
            audio_generator = stream.generator()
            requests1 = (speech.StreamingRecognizeRequest(audio_content=content)
                         for content in audio_generator)
            responses = client.streaming_recognize(streaming_config, requests1)

            # Get user's vocal response as text
            user_message_text = listen_print_loop(responses)

        messages.append({"role": "user", "content": user_message_text})
        summary_builder.add_message("user", user_message_text)

        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=messages,
        )
        assistant_message_text = response["choices"][0]["message"]["content"]

        # Output assistant’s message
        print("Assistant: ", assistant_message_text)

        messages.append(
            {'role': 'assistant', 'content': assistant_message_text})
        summary_builder.add_message("assistant", assistant_message_text)
        synthesize_speech(assistant_message_text)

    final_summary = summary_builder.generate_summary()
    #print("\nConversation Summary:")
    # print(final_summary)

    redirect_message = "Thank you. Your call is now being directed to an operator. Please hold."
    print("\nAssistant: ", redirect_message)
    synthesize_speech(redirect_message)

    condensed_summary = summarize_with_openai(final_summary)
    print("\nCondensed Summary:")
    print(condensed_summary)

    data1 = {"agentSummary": condensed_summary}
    response = requests.put(url, json=data1)
    if response.status_code == 200:
        pass
    else:
        print(f"Request failed with status code {response.status_code}")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        temp = sys.argv[1]
        run_conversation(temp)
    else:
        print("Error: Please provide the 'temp' argument.")
