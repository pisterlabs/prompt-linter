#!/usr/bin/env python

import rospy
import pyaudio
import webrtcvad
import numpy as np
import collections
import openai
import os
import requests
from pydub import AudioSegment
import noisereduce as nr
import tempfile
from qt_interaction.srv import TextToSpeech
from qt_interaction.srv import RasaService
from std_msgs.msg import String

rasa_endpoint = "http://localhost:5005/webhooks/rest/webhook" # Rasa server endpoint
openai.api_key = os.environ.get("OPENAI_API_KEY") # OpenAI API key from environment variable

def audio_to_text(audio_data, sample_rate=16000, channels=1, bit_depth=16):
    """
    Function takes in audio data, sample rate, number of channels, and bit depth,
    and returns the transcribed text. Temp file created due to OpenAI API requirement
    """
    audio_segment = AudioSegment(
        audio_data,
        frame_rate=sample_rate,
        sample_width=bit_depth // 8,
        channels=channels
    )

    audio_np = np.array(audio_segment.get_array_of_samples())
    reduced_noise = nr.reduce_noise(y=audio_np, sr=sample_rate)

    noise_reduced_segment = AudioSegment(
        reduced_noise.tobytes(),
        frame_rate=sample_rate,
        sample_width=bit_depth // 8,
        channels=channels
    )

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as temp_file:
        noise_reduced_segment.export(temp_file.name, format="wav")

        with open(temp_file.name, "rb") as audio_file:
            transcript = openai.Audio.transcribe("whisper-1", audio_file, language="en")
            return transcript['text']

def recognize_speech():
    """
    Function that uses the WebRTC Voice Activity Detector to detect speech in audio
    chunks, and returns the transcribed text if the audio chunk is long enough.
    """
    vad = webrtcvad.Vad()
    vad.set_mode(3)

    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 16000
    CHUNK_SIZE = 480
    SILENCE_CHUNKS_THRESHOLD = 20
    MINIMUM_AUDIO_DURATION_MS = 100

    pa = pyaudio.PyAudio()
    stream = pa.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK_SIZE)

    ring_buffer = collections.deque()
    num_silence_chunks = 0

    while not rospy.is_shutdown():
        chunk = np.frombuffer(stream.read(CHUNK_SIZE), dtype=np.int16)
        is_speech = vad.is_speech(chunk.tobytes(), RATE)

        if is_speech:
            ring_buffer.append(chunk)
            num_silence_chunks = 0
        else:
            num_silence_chunks += 1

            if len(ring_buffer) > 0 and num_silence_chunks >= SILENCE_CHUNKS_THRESHOLD:
                audio_data = b"".join([c.tobytes() for c in ring_buffer])
                audio_duration_ms = (len(ring_buffer) * CHUNK_SIZE) / RATE * 1000

                if audio_duration_ms >= MINIMUM_AUDIO_DURATION_MS:
                    transcribed_text = audio_to_text(audio_data)
                    if transcribed_text.strip():
                        print("Transcribed text:", transcribed_text)
                        rasa_service = rospy.ServiceProxy('rasa_service', RasaService)
                        response = rasa_service(transcribed_text)
                        return response.response

                ring_buffer.clear()

    stream.stop_stream()
    stream.close()
    pa.terminate()

def client_node():
    """
    ROS client that continuously listens for audio input, transcripts the audio,
    and sends the transcribed text to the text-to-speech service.
    """
    rospy.init_node('client_node', anonymous=True)
    rospy.wait_for_service('text_to_speech')
    text_to_speech = rospy.ServiceProxy('text_to_speech', TextToSpeech)


    while not rospy.is_shutdown():
        text = recognize_speech()
        if text:
            try:
                response = text_to_speech(text)
                if response.success:
                    rospy.loginfo("Text-to-speech service completed successfully.")
                else:
                    rospy.logerr("Text-to-speech service failed.")
            except rospy.ServiceException as e:
                rospy.logerr("Text-to-speech service call failed: %s", e)

if __name__ == '__main__':
    try:
        client_node()
    except rospy.ROSInterruptException:
        pass

