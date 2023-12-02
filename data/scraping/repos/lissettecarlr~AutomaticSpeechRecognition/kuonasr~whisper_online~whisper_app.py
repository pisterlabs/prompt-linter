# 使用whisper的方式进行语音转文字
# 官方文档：https://platform.openai.com/docs/guides/speech-to-text
import langid
from openai import OpenAI

class Whispers():
    def __init__(self,api_key,base_url):
        self.client = OpenAI(api_key=api_key,base_url=base_url)

    def infer(self,audio_path):
        try:
            with open(audio_path, 'rb') as audio_file:
                #response = openai.Audio.transcribe('whisper-1', audio_file,language="zh")
                transcript = self.client.audio.transcriptions.create(
                            model="whisper-1", 
                            file=audio_file, 
                            response_format="text",
                            language="zh"
                )
        except Exception as e:
            raise ValueError("语音转换失败，错误信息：{}".format(e))
        if(self.filter(transcript) ==False):
            raise ValueError("识别结果不符合要求，已经被过滤。{}".format(transcript))
        return transcript

    # 过滤 单个字符可能也会返回false
    def filter(self,text):
        lang, prob = langid.classify(text)
        if lang == 'zh' or lang == 'en':
            return True
        else:
            return False 
        

if __name__ == '__main__':
    key = ""
    url = ""
    audio_path = "../audio/asr_example.wav"
    service = Whispers(key,url)
    result = service.infer(audio_path) 
    print(result)        