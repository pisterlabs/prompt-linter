from threading import Thread

import openai
import pyaudio
import wave


def rating_responder(rating):
    if rating == None:
        pass
    if rating == 1:
        play_audio("../laughs_and_boos_and_cheers/1.wav")
    if rating == 2:
        play_audio("../laughs_and_boos_and_cheers/2.wav")
    if rating == 3:
        play_audio("../laughs_and_boos_and_cheers/3.wav")
    if rating == 4:
        play_audio("../laughs_and_boos_and_cheers/4.wav")
    if rating == 5:
        play_audio("../laughs_and_boos_and_cheers/5.wav")
    if rating == 6:
        play_audio("../laughs_and_boos_and_cheers/6.wav")
    if rating == 7:
        play_audio("../laughs_and_boos_and_cheers/7.wav")
    if rating == 8:
        play_audio("../laughs_and_boos_and_cheers/8.wav")
    if rating == 9:
        play_audio("../laughs_and_boos_and_cheers/9.wav")
    if rating == 10:
        play_audio("../laughs_and_boos_and_cheers/10.wav")


def play_audio(wav_file_path):
    # Open the WAV file
    wf = wave.open(wav_file_path, 'rb')

# Initialize PyAudio
    p = pyaudio.PyAudio()

    # Define callback to stream audio chunks
    def callback(in_data, frame_count, time_info, status):
        data = wf.readframes(frame_count)
        return (data, pyaudio.paContinue)



    # Open a new audio stream
    stream = p.open(format=p.get_format_from_width(wf.getsampwidth()),
                    channels=wf.getnchannels(),
                    rate=wf.getframerate(),
                    output=True,
                    stream_callback=callback)

    # Start the stream
    stream.start_stream()

    # Keep the stream open until all data is played
    while stream.is_active():
        # You can include a time.sleep here if you want to, for example:
        # time.sleep(0.1)
        pass

    # Stop and close the stream
    stream.stop_stream()
    stream.close()

    # Terminate PyAudio
    p.terminate()