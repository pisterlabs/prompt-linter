import os
import tempfile
import subprocess
import threading
import queue
import time
from playsound import playsound
import openai

openai.api_key = os.environ["OPENAI_API_KEY"]

# La fuente de microfono viene de Android a través de AudioRelay

# Cola para almacenar nombres de archivos de audio temporales
audio_queue = queue.Queue()

def trans():
    while True:
        audio = audio_queue.get()
        if audio is None:
            break

        # transcribe audio
        #subprocess.run(["whisper", audio, "--language", "Spanish", "--model", "small", "--output_dir", "/tmp"])
        print(audio)
        #playsound(audio)
        audio_file= open(audio, "rb")
        transcript_es = openai.Audio.transcribe(
            file = audio_file,
            model = "whisper-1",
            response_format="text",
            language="es"
        )

        # remove temporary audio file
        os.remove(audio)

        # type text to terminal, in background
        print(transcript_es)
        
        '''
        tmp_file_path = '/tmp/tmp.txt'
        if os.path.exists(tmp_file_path):
            with open(tmp_file_path, 'r') as f:
                text = f.read()
            
            if text != "Thanks for watching!":
                subprocess.run(["xdotool", "type", "--clearmodifiers", "--file", tmp_file_path])
                subprocess.run(["xdotool", "key", "space"])
                os.remove(tmp_file_path)
        '''
                
# record audio in background
def record():
    while True:
        # make temporary files to hold audio
        tmp = tempfile.mktemp()+".mp3"

        # listen to mic
        # Listen to mic. The `&` lets it operate in the background.
        # The `1 0.2 3%` part of the sox rec command trims 1 segment of silence from
        # the beginning longer than 0.2 seconds and longer than 3% of the volume level.
        # The final `1 2.0 1%` part tells it to trim 1 segment of silence from the end.
        # It stops recording after 2.0 seconds of silence. Change to 5% or more with
        # poor recording equipment and noisy environments.
        subprocess.run(["rec", "-c", "1", "-r", "22050", "-t", "mp3", tmp, "silence", "1", "0.05", "3%", "1", "0.5", "4%"])

        # add temporary audio file name to transcription queue
        audio_queue.put(tmp)

        # sleep for a short while to prevent high CPU usage
        time.sleep(0.01)

# run audio transcription handler
t = threading.Thread(target=trans)
t.start()

# start recording
record()