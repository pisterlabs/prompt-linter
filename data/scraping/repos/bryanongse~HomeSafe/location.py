from tarfile import ExtractError
import cohere
import pprint
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import pyttsx3
import speech_recognition as sr
from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import gtts
from playsound import playsound
import random
import sounddevice as sd
from scipy.io.wavfile import write
from io import BytesIO
import time
import wavio
from detoxify import Detoxify


##### Speech to Text Var
r = sr.Recognizer()
engine = pyttsx3.init()

##### Recording Var
freq = 44100
duration = 5

##### Global Variables
flag = 0
score = 0
scenerioNo = 2
threshold = 1.9
continueNo = 7
successNo = 4

def record():
    #playsound("ding.mp3")

    recording = sd.rec(int(duration * freq),
                       samplerate=freq, channels=2)
    sd.wait()

    num = random.randint(0, 1000)
    name = "recording" + str(num) + ".wav"
    wavio.write("./sound/"+name, recording, freq, sampwidth=2)

    text = speechToText("./sound/"+name)  # filename is the directory where the user's file is downloaded ###########################

    return text

def speechToText(filename):
    with sr.AudioFile(filename) as source:
        audio_data = r.record(source)
        text = r.recognize_google(audio_data)
        return text


def cohere(streamTxt):
    api_key = 'ubrYIiK7ptgJ6UdrWTMxVYBsastwkbp7ulstaeHE'
    co = cohere.Client(api_key)

    example_data = [
        ("People's Park", "Violent Crime Reported at People's Park - Please note this message may contain information that some may find upsetting"),
        ("Tallmadge", "Car 24 we're east on Tallmadge approaching RT 8 South"),
        ("Telegraph and Durant", "AVOID THE AREA of Telegraph and Durant"),
        ("5024 Washburn Avenue South", "Squad 530 to 5024 Washburn Avenue South. Female screaming behind the building."),
        ("Vidar Street", "Pretty large crowd of juvvies heading down Vidar Street toward Kenton sports ground. Look to be drinking, we're coming up behind them now."),
        ("Ervay Street", "Ervay Street is completely blocked with pedestrians. Completely out of control."),
        ("Murphy Park", "Dispatch, suspect vehicle is at Murphy Park, no suspect. Additional units requested"),
        ("17th Street", "Suspect seen heading north on 17th street. 2015 Dodge Durango, silver, license plate number"),
        ("17 Ware Street", "I just had a, uh, older woman standing here and she had noticed two gentlemen trying to get in a house at that number 17 Ware Street."),
        ("Rose", "Ford Fusion heading westbound on Rose, last seen from my location.")
    ]

    class cohereExtractor():
        def __init__(self, examples, example_labels, labels, task_desciption, example_prompt):
            self.examples = examples
            self.example_labels = example_labels
            self.labels = labels
            self.task_desciption = task_desciption
            self.example_prompt = example_prompt

        def make_prompt(self, example):
            examples = self.examples + [example]
            labels = self.example_labels + [""]
            return (self.task_desciption +
                    "\n---\n".join( [examples[i] + "\n" +
                                    self.example_prompt + 
                                    labels[i] for i in range(len(examples))]))

        def extract(self, example):
            extraction = co.generate(
                model='large',
                prompt=self.make_prompt(example),
                max_tokens=10,
                temperature=0.1,
                stop_sequences=["\n"])
            return(extraction.generations[0].text[:-1])
        

    cohereLocationExtractor = cohereExtractor([e[1] for e in example_data],
                                                [e[0] for e in example_data],
                                                [],
                                                "",
                                                "location from crime report:")
                                    
    test_data = [
        "Suspect spotted going westbound on 42nd Street. Heading to the location now.",
        "Avoid the area of Channing and Bowditch.",
        "Car seen speeding northbound on Jeppe Street.",
        "Large group of people getting out of control seen around the area of Piedmont and Durant",
        "Chasing an individual speeding in a large truck heading east on 20th Street",
        "Additional units required on the 2600 block of Butte.",
        "Adrians kickback is getting out of control I need backup on Ocean Ave.",
        "Group of Berkeley students are trying to get into a hotel at 780 Mission St"
    ]

    results = []
    # for text in tqdm(test_data):
    #     try:
    #         extracted_text = cohereLocationExtractor.extract(text)
    #         results.append(extracted_text)
    #     except Exception as e:
    #         print('ERROR: ', e)
    with ThreadPoolExecutor(max_workers=8) as executor:
        for i in executor.map(cohereLocationExtractor.extract, test_data):
            results.append(str(i).strip())

    pprint.pprint(results)

def main():
    text = record()
    cohere(text)
main()
