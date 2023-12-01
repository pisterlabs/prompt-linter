import openai
import os

# Set OpenAI API key
openai.api_key = os.getenv("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")

# Load text from file
with open("C:\Davidlocal\PMD_AI_TRANNING2\exampleconversation2.txt", "r", encoding="utf-8") as f:
    input_text = f.read()

# Fine-tune the GPT-3 model on the input text
response = openai.Completion.create(
    engine="text-davinci-002",
    prompt=(f"Fine-tune the following text:\n\n{input_text}\n\n\nWith the following continuation:\n\n"),
    max_tokens=1024,
    n=1,
    stop=None,
    temperature=0.5,
    frequency_penalty=0,
    presence_penalty=0
)

# Print the fine-tuned text
print(response.choices[0].text)
