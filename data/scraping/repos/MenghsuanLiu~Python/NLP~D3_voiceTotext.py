# %%
# 錄音程式
import sounddevice                      # 匯入sounddevice
from scipy.io.wavfile import write      # 匯入 scipy.io.wavfile.write

fs= 44100                               # 設定聲音採樣頻率 DVD中的採樣頻率
second =  int(input("輸入要錄音的秒數: ")) # 輸入要錄音的秒數 要打整數
print("Recording.....")
                                        # https://pypi.org/project/sounddevice/0.3.2/
record_voice = sounddevice.rec(
    int ( second * fs ) ,               # 錄音秒數 * 採樣頻率
    samplerate = fs ,                   # 採樣頻率
    channels = 2 )                      # 錄音兩個聲道
sounddevice.wait()                      # 等待錄音完成 才會執行到下一行, Ctrl+C可以中斷錄音
write("./Data/out.wav",fs,record_voice)        # 寫入檔案
print("完成, 輸出到 out.wav")             # 完成輸出到 out.wav

# %%
# 錄咅轉文字
import soundfile
import speech_recognition as sr
data, samplerate = soundfile.read('./Data/out.wav')        # 讀取out.wav 檔案

r = sr.Recognizer()  # 預設辨識英文 初始化

# wave 檔案 轉檔 PCM_16
soundfile.write('./Data/new.wav', data, samplerate, subtype='PCM_16')
with sr.WavFile("./Data/new.wav") as source:          # 讀取wav檔
    audio = r.record(source)

# 辨聲音檔案　辨識成中文輸出,語系使用ISO 639-1
try:
    str1= r.recognize_google(audio,language="zh-TW")    # 使用Google的服務
    print("辨識後的文字: " + str1)

except LookupError:
    print("錯誤:Could not understand audio")
# %%
# 即時語音轉文字
import speech_recognition as sr

r = sr.Recognizer()                                           # 預設辨識英文

print(sr.Microphone.list_microphone_names())                  # 列出所有的麥克風
print("請說話，結束時，按下Ctrl+C  就可以辨識語音")
#source = sr.Microphone(device_index=0)                       # 麥克風設定 0 內定
microphone = sr.Microphone()                                  # 麥克風設定 0 內定
with microphone as source:
    r.adjust_for_ambient_noise(source)                      # 調整麥克風的雜訊
    audio = r.listen(source)                                # 錄製的語音


# str1 =r.recognize_google(audio, key="GOOGLE_SPEECH_RECOGNITION_API_KEY", language="zh-TW")
str1 = r.recognize_google(audio, language="zh-TW")            # 使用Google的服務
print("辨識後的文字: " +str1)

# %%
# 即時語音轉文字(用語音控制結束)
import speech_recognition as sr

r = sr.Recognizer()                                           # 預設辨識英文

print(sr.Microphone.list_microphone_names())                  # 列出所有的麥克風
print("說一段話，並停3秒 ， 就可以辨識語音，如果要結束，就說「離開」")
#source = sr.Microphone(device_index=0)                       # 麥克風設定 0 內定
microphone = sr.Microphone()                                  # 麥克風設定 0 內定
while True:
    with microphone as source:
        r.adjust_for_ambient_noise(source)                      # 調整麥克風的雜訊
        audio = r.listen(source)                                # 錄製的語音


    # str1 =r.recognize_google(audio, key="GOOGLE_SPEECH_RECOGNITION_API_KEY", language="zh-TW")
    # str1 = r.recognize_google(audio, language="zh-TW")            # 使用Google的服務
    str1 = r.recognize_whisper(audio)            # 使用OpenAI的服務
    print("辨識後的文字: " +str1)
    if str1 == "離開" or str1.find("離開") >=0  or str1.find("再見") >=0  or str1.find("關閉") >=0:
        break
# %%
# 使用open ai做
import openai
import json

try:
    # Open the JSON file for reading
    with open("./Config/api.json", "r") as json_file:
        data = json.load(json_file)
        openai.api_key = data["openai"]
except FileNotFoundError:
    print(f"The file api.json does not exist.")

audio_file= open("./Data/out.wav", "rb")
responce = openai.Audio.transcribe("whisper-1", audio_file)
print(responce)
print(responce["text"])




# %%
