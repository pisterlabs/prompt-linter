# %%
import openai
import playsound
import __init__


# %%
def tts_response(text, file_path="./speech/speech.mp3"):
    response = openai.audio.speech.create(
      model="tts-1",
      voice="nova",
      input=text
    )
    response.stream_to_file(file_path)
    playsound.playsound(file_path)
# %%
if __name__ == "__main__":
    text = "回答问题、聊天、提供信息、制定计划等。回答问题、聊天、提供信息、制定计划等。"
    tts_response(text)
