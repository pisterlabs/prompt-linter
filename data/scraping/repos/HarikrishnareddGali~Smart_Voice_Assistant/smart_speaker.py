import os
import openai
import pyaudio
import speech_recognition as sr
from gtts import gTTS
from pydub import AudioSegment
from pydub.playback import play
import apa102
import threading
from gpiozero import LED
try:
    import queue as Queue
except ImportError:
    import Queue as Queue
from alexa_led_pattern import AlexaLedPattern
from modelsCollection.models import Models
import json
 
 
# load pixels Class
class Pixels:
    PIXELS_N = 12
 
    def __init__(self, pattern=AlexaLedPattern):
        self.pattern = pattern(show=self.show)
        self.dev = apa102.APA102(num_led=self.PIXELS_N)
        self.power = LED(5)
        self.power.on()
        self.queue = Queue.Queue()
        self.thread = threading.Thread(target=self._run)
        self.thread.daemon = True
        self.thread.start()
        self.last_direction = None
 
    def wakeup(self, direction=0):
        self.last_direction = direction
        def f():
            self.pattern.wakeup(direction)
 
        self.put(f)
 
    def listen(self):
        if self.last_direction:
            def f():
                self.pattern.wakeup(self.last_direction)
            self.put(f)
        else:
            self.put(self.pattern.listen)
 
    def think(self):
        self.put(self.pattern.think)
 
    def speak(self):
        self.put(self.pattern.speak)
 
    def off(self):
        self.put(self.pattern.off)
 
    def put(self, func):
        self.pattern.stop = True
        self.queue.put(func)
 
    def _run(self):
        while True:
            func = self.queue.get()
            self.pattern.stop = False
            func()
 
    def show(self, data):
        for i in range(self.PIXELS_N):
            self.dev.set_pixel(i, int(data[4*i + 1]), int(data[4*i + 2]), int(data[4*i + 3]))
 
        self.dev.show()
 
pixels = Pixels()
 
 
# settings and keys
openai.api_key = os.environ.get('OPENAI_API_KEY')
model_engine = "text-davinci-003"
 
def recognize_speech():
    # obtain audio from the microphone
    r = sr.Recognizer()
    with sr.Microphone() as source:
        print("Waiting for wake word...")
        while True:
            try:
                r.adjust_for_ambient_noise(source)
                audio_stream = r.listen(source)
                # recognize speech using Google Speech Recognition
                try:
                    # convert the audio to text
                    print("Google Speech Recognition thinks you said " + r.recognize_google(audio_stream,language = "de-DE"))
                    speech = r.recognize_google(audio_stream,language = "de-DE")
                    if ("Pixel" not in speech) and ("pixel" not in speech):
                        # the wake word was not detected in the speech
                        print("Wake word not detected in the speech")
   						# Close the current microphone object
                        return False
                    else:
                        # the wake word was detected in the speech
                        print("Found wake word!")
                        # wake up the display
                        pixels.wakeup()
                        return True
                except sr.UnknownValueError:
                    print("Google Speech Recognition could not understand audio")
                    print("Waiting for wake word...")
                    return False
                except sr.RequestError as e:
                    print("Could not request results from Google Speech Recognition service; {0}".format(e))
                    print("Waiting for wake word...")
                    return False
            except KeyboardInterrupt:
                print("Interrupted by User Keyboard")
                break
 
def speech():
    # obtain audio from the microphone
    r = sr.Recognizer()
    with sr.Microphone() as source:
        print("Waiting for user to speak...")
        while True:
            try:
                r.adjust_for_ambient_noise(source)
                audio_stream = r.listen(source)
                # recognize speech using Google Speech Recognition
                try:
                    # convert the audio to text
                    print("Google Speech Recognition thinks you said " + r.recognize_google(audio_stream,language = "de-DE"))
                    speech = r.recognize_google(audio_stream,language = "de-DE")
                    # wake up thinking LEDs
                    pixels.think()
                    return speech
                except sr.UnknownValueError:
                    print("Google Speech Recognition could not understand audio")
                    pixels.off()
                    print("Waiting for user to speak...")
                    continue
                except sr.RequestError as e:
                    print("Could not request results from Google Speech Recognition service; {0}".format(e))
                    pixels.off()
                    print("Waiting for user to speak...")
                    continue
            except KeyboardInterrupt:
                print("Interrupted by User Keyboard")
                break
            
 
def load_config(filename):
    with open(filename, 'r') as file:
        return json.load(file)

def chatgpt_response(prompt):
    models_instance = Models()
    
    config = load_config("/home/pi/website_voiceassistant/config/config_openaimain.json")
    # Extracting arguments from the config file
    args = config["args"]
    # Adding the prompt to the arguments
    args.append(prompt)

    # Use reflection to get the method by name
    model_function = getattr(models_instance, config["model_function"])
    result = model_function(*args)

    if result and 'answer' in result:
        return result['answer']
 
def generate_audio_file(text):
    # convert the text response from chatgpt to an audio file 
    audio = gTTS(text=text, lang='de', slow=False)
    # save the audio file
    audio.save("response.mp3")
 
def play_audio_file():
    song = AudioSegment.from_mp3("response.mp3")
    play(song)
 
def main():
    # run the program
    while True:
        if recognize_speech():
            prompt = speech()
            print(f"This is the prompt being sent to OpenAI: {prompt}")
            response = chatgpt_response(prompt)
            print(response)
            generate_audio_file(response)
            play_audio_file()
            pixels.off()
        else:
            print("Speech was not recognised")
            pixels.off()
 
if __name__ == "__main__":
    main()
