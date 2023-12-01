#uses old version of openai api

import openai
from docx import Document

# Set up OpenAI API credentials
openai.api_key = 'YOUR_API_KEY'

yourPrompt = input("Enter your prompt: ")

# Generate text using OpenAI
response = openai.Completion.create(
  engine='davinci',
  prompt=yourPrompt,
  max_tokens=100
)
generated_text = response.choices[0].text.strip()

# Create a new Word document
doc = Document()
doc.add_paragraph(generated_text)

# Save the document as a Word file
doc.save('output.docx')
