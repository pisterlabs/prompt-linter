import openai
import pyttsx3
import speech_recognition as sr

# OpenAIのAPIキーを取得する
openai.api_key = "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"

# 音声合成のエンジンを初期化する
engine = pyttsx3.init()

def transcribe_audio_to_text(filename):
    # 音声認識用のモジュールを初期化
    recongnizer = sr.Recognizer()
    
    # 引数として受け取った音声ファイルから，音声を読み込み．
    with sr.AudioFile(filename) as source:
        audio = recongnizer.record(source)
    try:
        # 音声を認識し，結果を返す．
        return recongnizer.recognize_google(audio)
    except:
        print("Skipping unknown error")
    
def generate_response(prompt):
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=prompt,
        max_tokens=4000,
        n=1, 
        stop=None,
        temperature=0.5
    )
    return response["choices"][0]["text"]

def speak_text(text):
    engine.say(text)
    engine.runAndWait()
    
def main():
    while True:
        # Userが"genius"と呼ぶまで待つ
        print("Say ''Genius to start recording your question...")
        with sr.Microphone() as source:
            recognizer = sr.Recognizer()
            audio = recognizer.listen(source)
            try:
                transcription = recognizer.recognize_google(audio)
                if transcription.lower() == "genius":
                    # Record audio
                    filename = "input.wav"
                    print("Say your question...")
                    with sr.Microphone() as source:
                        recognizer = sr.Recognizer()
                        source.pause_threshold = 1
                        audio = recognizer.listen(source, phrase_time_limit=None, timeout=None)
                        with open(filename, "wb") as f:
                            f.write(audio.get_wav_data())
                    
                    # transcribe audio to text
                    text = transcribe_audio_to_text(filename)
                    if text:
                        print(f"You Said: {text}")
                        
                        # generate response
                        response = generate_response(text)
                        print(f"GPT-3 says: {response}")
                        
                        # Read response using text to speech
                        speak_text(response)
            except Exception as e:
                print("An error occurred: {}".format(e))
                
if __name__ == "__main__":
    main()