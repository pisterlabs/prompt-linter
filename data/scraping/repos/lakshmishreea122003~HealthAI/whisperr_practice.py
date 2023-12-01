# import openai
# audio_file= open("C:/Users/Lakshmi/Downloads/output.mp3", "rb")
# transcript = openai.Audio.transcribe("whisper-1", audio_file)
# print(transcript)



from gtts import gTTS

language = 'en'
text= "My voice matters"
speech = gTTS(text = text, lang= language)
speech.save("D:\llm projects\SilentBridge-streamlit\pages\out.mp3")

