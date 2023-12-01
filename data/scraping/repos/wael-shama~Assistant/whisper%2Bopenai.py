import whisper
import openai
import os



model = whisper.load_model("base")

result = model.transcribe("./audios/appointment.mp3")
print(result["text"])



# Set the OpenAI API key
print(os.environ['OPENAI_API_KEY'])
openai.api_key = os.environ['OPENAI_API_KEY']

# Generate text using OpenAI's GPT-3
prompt = "Write a short story about a robot who falls in love with a human"
response = openai.Completion.create(
    engine="text-davinci-002",
    prompt=prompt,
    max_tokens=1024,
    n=1,
    stop=None,
    temperature=0.5,
)

# Print the generated text
print(response.choices[0].text)

