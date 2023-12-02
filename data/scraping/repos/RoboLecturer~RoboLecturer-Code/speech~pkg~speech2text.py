# pip install SpeechRecognition pyaudio
import speech_recognition as sr
# import openai
r = sr.Recognizer()

def Sphinx(audio):
    # recognize speech using Sphinx
    try:
        return r.recognize_sphinx(audio)
    except sr.UnknownValueError:
        print("Sphinx could not understand audio")
    except sr.RequestError as e:
        print("Sphinx error; {0}".format(e))

def GSR(audio):
    # recognize speech using Google Speech Recognition
    try:
        # for testing purposes, we're just using the default API key
        # to use another API key, use `r.recognize_google(audio, key="GOOGLE_SPEECH_RECOGNITION_API_KEY")`
        # instead of `r.recognize_google(audio)`
        return r.recognize_google(audio)
    except sr.UnknownValueError:
        print("Google Speech Recognition could not understand audio")
    except sr.RequestError as e:
        print("Could not request results from Google Speech Recognition service; {0}".format(e))

def whisper(audio, online=True):
    # Use openai online API, else offline pre-trained model
    if online:
        openai.Audio.transcribe("whisper-1", audio)
    else:
        # recognize speech using whisper
            try:
                return r.recognize_whisper(audio, language="english")
            except sr.UnknownValueError:
                print("Whisper could not understand audio")
            except sr.RequestError as e:
                print("Could not request results from Whisper")


# Test with mic
def runS2T(index):
    with sr.Microphone(device_index=index) as source:
        print("Say something!")
        if source.stream is None:
            print("NONE")
        mic_input = r.listen(source, 35, 3)
        print("done listening")
        # text = whisper(mic_input)
        text = GSR(mic_input)
    return text
# print(runS2T(0))
