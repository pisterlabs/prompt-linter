import whisper
import openai

# openai.api_key = 'enter API key'

model = whisper.load_model("base")
audio = "audio.mp3" # save audio files as audio.mp3 or change this line
result = model.transcribe(audio)

with open("transcription.txt", "w", encoding="utf-8") as txt:
    txt.write(result["text"])

def process_prompt(prompt):
    # Make an API call to ChatGPT
    response = openai.completions.create(
        engine="text-davinci-003",
        prompt=prompt,
        max_tokens=500, # change response tokens as needed
        temperature=0.7,
        n=1,
        stop=None,
        timeout=None,
    )

    return response.choices[0].text.strip()

generated_response = process_prompt(result)
print(generated_response)
