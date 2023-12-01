import requests
import speech_recognition as sr
import threading
import queue
import keyboard
from openai import OpenAI
import urllib3

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


# Multi-Thread_01 controling input audio
class SpeechToTextThread(threading.Thread):
    def __init__(self, audio_queue, flag):
        threading.Thread.__init__(self)
        self.recognizer = sr.Recognizer()
        self.audio_queue = audio_queue
        self.flag = flag

    def run(self):
        with sr.Microphone() as source:
            self.recognizer.adjust_for_ambient_noise(source)
            while not self.flag.is_set():
                try:
                    print("I'm listening..Umm...")
                    audio = self.recognizer.listen(source, timeout=None, phrase_time_limit=10)
                    self.audio_queue.put(audio)
                    
                except sr.UnknownValueError:
                    print("I didn't understand your words...Come again Plz?")
                    print()
                    pass  # Ignore if the audio is not recognized
                except sr.RequestError as e:
                    print(f"Google Cloud Speech-to-Text request failed: {e}")

def main():
    audio_queue = queue.Queue()
    flag = threading.Event()
    speech_thread = SpeechToTextThread(audio_queue, flag)
    answer = "[Base] ChatGPT, which stands for Chat Generative Pre-trained Transformer, is a large language model-based chatbot developed by OpenAI and launched on November 30, 2022, that enables users to refine and steer a conversation towards a desired length, format, style, level of detail, and language. Successive prompts and replies, known as prompt engineering, are considered at each conversation stage as a context. [Answer] "

    try:
        # Start the speech-to-text thread
        speech_thread.start()
        
        url_for_deepl = 'https://api-free.deepl.com/v2/translate'

        # Multi_Thread_Main requesting origin audio data to GOOGLE & printing configuration
        while not flag.is_set():
            try:
                # Get audio from the queue
                audio = audio_queue.get(block=True, timeout=1)
                text = speech_thread.recognizer.recognize_google_cloud(
                    audio,
                    credentials_json='credential.json',
                    language='ko-KR',
                )
                params = {'auth_key' : 'auth_key', 'text' : text, 'source_lang' : 'KO', "target_lang": 'EN' }
                result = requests.post(url_for_deepl, data=params, verify=False)
                print(f"Transcription: {text}")
                print(result.json()['translations'][0]["text"])

                answer += result.json()['translations'][0]["text"]
            except queue.Empty:
                pass  # Queue is empty, no audio available
            except sr.UnknownValueError:
                print("I didn't understand your words...Come again Plz?")
                print()
                pass  # Ignore if the audio is not recognized
            except sr.RequestError as e:
                print(f"Google Cloud Speech-to-Text request failed: {e}")
            
    except KeyboardInterrupt:
        # Stop the speech-to-text thread when the program is interrupted
        print("Exiting Education..")
        flag.set()
        speech_thread.join()

    print(answer)

    # f = open('gpt_key.txt', 'r')
    # api_key = f.read()
    api_key = "api_key"

    client = OpenAI(api_key = api_key)


    completion = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "system", "content": "The Structured Feedback Analyzer now has an enhanced function. It will work with a base sentence provided once ([Base]) and multiple incomplete sentences ([Answer]). Each [Answer] will end with a flag indicating whether it is complete (<True>) or incomplete (<False>). The GPT's task is to assess the correctness of each [Answer] up to the point it is given. The focus is on analyzing the grammatical accuracy and contextual relevance of the [Answer] in relation to the [Base]. This GPT will not only compare the [Answer] to the [Base] but also evaluate the correctness of the [Answer] as a standalone sentence. The feedback provided will be concise, focusing on the correctness of the [Answer] up to the point it is given, without speculating about the missing parts. This structured approach will help users understand the accuracy and relevance of their answers in relation to the base sentence."},
        {"role": "user", "content": answer}
    ]
    )

    print(completion.choices[0].message.content)

    

if __name__ == "__main__":
    main()