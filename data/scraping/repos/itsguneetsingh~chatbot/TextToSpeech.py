# using ElevenLabs for TTS

import constants
from  elevenlabs import set_api_key, generate, play
# from langchain import *
from langchain.callbacks.base import BaseCallbackHandler

set_api_key(constants.ELEVENLABS_API_KEY)

def tts(text):
    audio_bytes = b"".join(list(generate(
        text=text,
        voice="Callum",
        model="eleven_multilingual_v1",
        stream=True)))
    
    play(audio_bytes)

class TTSCallbackHandler(BaseCallbackHandler):
    def __init__(self) -> None:
        self.content: str = ""

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.content += token
            
    def on_llm_end(self, response: dict, **kwargs : any) -> any:
        print(self.content)
        tts(self.content)
        self.content = ""
        # return super().on_llm_end(self.content, **kwargs)


# Might come in handy but forgot what to use this for
    # def on_llm_new_token(self, token: str, **kwargs: any) -> None:
    #     self.content += token
    #     if "Final Answer" in self.content:
    #         # now we're in the final answer section, but don't print yet
    #         self.final_answer = True
    #         self.content = ""
    #     if self.final_answer:
    #         if '"action_input": "' in self.content:
    #             if token not in ["}"]:
    #                 sys.stdout.write(token)  # equal to `print(token, end="")`
    #                 sys.stdout.flush()



# To get a list of voices:

# import requests

# url = "https://api.elevenlabs.io/v1/voices"

# headers = {
#   "Accept": "application/json",
#   "xi-api-key": constants.ELEVENLABS_API_KEY
# }

# response = requests.get(url, headers=headers)

# print(response.text)