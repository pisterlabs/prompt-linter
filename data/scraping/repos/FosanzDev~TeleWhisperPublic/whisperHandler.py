import openai
import os
# import messages

class Whisperer:

    def __init__(self, apiKey, lang='auto'):
        self.lang = lang
        self.apiKey = apiKey

    async def setLang(self, lang):
        # If the language is auto, set it to empty
        if lang == 'auto':
            self.lang = ""
            return 0
        
        # If the language is not in the list, return 1
        # If the language is in the list, set it and return 0
        # for i in messages.LANGUAGES:
        #     if i == lang:
        #         self.lang = lang
        #         return 0
            
        return 1

    async def whisp(self, path):
        openai.api_key = self.apiKey
        
        __path__ = os.path.dirname(path)
        audio = open(path, "rb")

        transcript = await openai.Audio.atranscribe("whisper-1", audio, language= (None if self.lang == 'auto' else self.lang), responseFormat="text");

        return transcript["text"] # type: ignore