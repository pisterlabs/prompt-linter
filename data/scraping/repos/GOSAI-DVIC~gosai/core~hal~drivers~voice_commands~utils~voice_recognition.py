
import whisper
#import openai #try to use openai api to translate audio as an audio file 

class SpeechToText : 
    def __init__(self):
        self.device = "cpu"
        self.model = whisper.load_model("small.en", device = self.device)
        

    def translate(self,audio):
        transcript = whisper.translate(self.model, audio, language = "ENGLISH", fp16 = False)
        return transcript["text"]


 
   
    