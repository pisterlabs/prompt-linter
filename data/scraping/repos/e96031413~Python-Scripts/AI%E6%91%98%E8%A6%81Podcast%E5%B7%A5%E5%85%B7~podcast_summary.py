import whisper
import openai
import os

openai.api_key = "XXXXXXXXXXXXXXXX" 

# input audio file name
audio_file = "C://Users/e96031413/Desktop/test.mp3"

# load the model and transcribe the audio
# Set the argument in model.transcribe() accordingly.
model = whisper.load_model("base") # base
# result = model.transcribe(audio_file, fp16=False, language="English", verbose=True)
result = model.transcribe(audio_file, language="English", verbose=True)

# extract the text and language information
text = result["text"]
language = result["language"]

# print the text and language information to the console
print("Text:\n\n", text)
print("Language: ", language)

response = openai.ChatCompletion.create(
  model="gpt-3.5-turbo",
  messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": f"提供以下文字之繁體中文摘要，並以重點整理方式呈現，每個重點需有對應的描述與舉例，且重點需是普羅大眾不會知道的事情，而非普遍知識。若有專有名詞，請特別標註其原文，且內容須流暢通順，不得有贅字冗詞，以台灣地區的用語呈現：{text}"}
    ]
)
ans = response.to_dict()['choices'][0]['message']['content']
print(ans)

# create the output text file name based on the input mp3 file name
text_file = os.path.splitext(audio_file)[0] + ".txt"

# write the text, language, and summary to the output text file
with open(text_file, "w") as f:
    f.write(f"Text:\n\n{text}\n\nLanguage: {language}")
    f.write(f"\n\nSummary:\n\n{ans}")