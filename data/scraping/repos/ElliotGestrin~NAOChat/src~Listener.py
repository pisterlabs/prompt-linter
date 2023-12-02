import openai
import speech_recognition as sr
import sounddevice
from io import BytesIO

class Listener():
    def __init__(self, language: str="en", default_mic: bool=True, use_whisper: bool = False):
        """
        Creates a Listener object for speech-to-text

        Args:
            language (str): The ISO 639-1 code for the language used
            default_mic (bool): Wheter the default mic should be used. Otherwise user selects
            use_whisper (bool): If OpenAIs Whisper API should be used, worse in testing 
        """
        self.r = sr.Recognizer()
        self.language = language
        self.use_whisper = use_whisper
        if use_whisper and not openai.api_key:
            openai.api_key = open("openai.key").read().strip()
        if default_mic:
            self.mic = sr.Microphone()
        else:
            print("Please select a microphone from this list:")
            for i, m in enumerate(sr.Microphone.list_microphone_names()):
                print(f"   {i}: {m}")
            id = int(input("Select an index (-1 for default): "))
            if id == -1:
                self.mic = sr.Microphone()
            else:
                self.mic = sr.Microphone(device_index=id)
        with self.mic as mic:
            self.r.adjust_for_ambient_noise(mic)

    def recognize_whisper_api(self, audio_data: sr.AudioData):
        """
        Transcribes the given audio data into text via Whisper.
        Adapted from recognizer_instance.recognize_whisper_api

        Args:
            audio_data (st.AudioData): Audio to transcribe
        """
        if not isinstance(audio_data, sr.AudioData):
            raise ValueError("``audio_data`` must be an ``AudioData`` instance")

        wav_data = BytesIO(audio_data.get_wav_data())
        wav_data.name = "SpeechRecognition_aduio.wav"
        transcript = openai.Audio.transcribe("whisper-1", wav_data, api_key=openai.api_key,language=self.language)
        return transcript["text"]
    
    def __call__(self):
        """
        Listens to and returns a transcription of the next heard phrase
        """
        with self.mic as mic:
            audio = self.r.listen(mic)
        try:
            if self.use_whisper: return self.recognize_whisper_api(audio)
            else: return self.r.recognize_google(audio,language=self.language)
        except Exception:
            return ""
        
if __name__ == "__main__":
    from Chatter import Chatter
    name = "Pepper"
    swe_intro = f"Du är den mänskliga roboten {name}. En NAO-modellrobot byggd av Softbank och programmerad av FIA Robotics. Din uppgift är att hålla en intressant konversation med en grupp människor. Du får max svara med två meningar."
    eng_intro = f"You are the humanoid robot {name}. A NAO model robot built by Softbank and programmed by FIA Robotics. Your task is to hold an interesting conversation with a group of humans. You can at most answer with two sentences"
    chatter = Chatter(eng_intro,stream=True,name=name)
    listener = Listener(language="en",default_mic=False,use_whisper=False) # Change to 'sv' for english
    while(True):
        heard = listener()
        print(f"Heard: {heard}")
        if heard != "":
            response = chatter(heard)
            print(f"Response: {response}")