# %%
import speech_recognition as sr
import json
import os
import googletrans
# from pathlib import Path
# from openai import OpenAI
# %%
def recognize_speech_from_microphone():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("請開始說話...")
        audio = recognizer.listen(source)
        try:
            text = recognizer.recognize_google(audio, language="zh-TW") # 將語音轉換為文字
            print("辨識結果：", text)
            return text
        except sr.UnknownValueError:
            print("無法辨識語音")
            return None
        except sr.RequestError:
            print("無法連接到Google API")
            return None



if __name__ == "__main__":
    try:
        # Open the JSON file for reading
        with open("D:\\GitHub\\Python\\NLP\\Config\\api.json", "r") as json_file:
            data = json.load(json_file)
            os.environ["_BARD_API_KEY"] = data["bardkey"]
    except FileNotFoundError:
        print(f"The file api.json does not exist.")
    # Speech To Text
    input_text = recognize_speech_from_microphone()

    translator = googletrans.Translator()
    

    print('English:', translator.translate(input_text, dest = "en").text)
    print('Japanese:', translator.translate(input_text, dest = "ja").text)

    # client = OpenAI()
    # speech_file_path = Path(__file__).parent
    # response = client.audio.speech.create(
    #                                         model = "tts-1",
    #                                         voice = "alloy",
    #                                         input = translator.translate(input_text, dest = "ja").text )
    # response.stream_to_file(speech_file_path)

    # r = sr.Recognizer()                                           # 預設辨識英文

    # print(sr.Microphone.list_microphone_names())                  # 列出所有的麥克風
    # print("說一段話，並停3秒 ， 就可以辨識語音，如果要結束，就說「離開」")
    # #source = sr.Microphone(device_index=0)                       # 麥克風設定 0 內定
    # microphone = sr.Microphone()                                  # 麥克風設定 0 內定
    # while True:
    #     with microphone as source:
    #         r.adjust_for_ambient_noise(source)                      # 調整麥克風的雜訊
    #         audio = r.listen(source)                                # 錄製的語音


    #     # str1 =r.recognize_google(audio, key="GOOGLE_SPEECH_RECOGNITION_API_KEY", language="zh-TW")
    #     # str1 = r.recognize_google(audio, language="zh-TW")            # 使用Google的服務
    #     str1 = r.recognize_whisper(audio)            # 使用OpenAI的服務
    #     print("辨識後的文字: " +str1)
    #     if str1 == "離開" or str1.find("離開") >=0  or str1.find("再見") >=0  or str1.find("關閉") >=0:
    #         break

    # 使用機器翻譯將中文翻譯成英文
    # en_text = ts.google(str1, from_language='zh', to_language='en')
