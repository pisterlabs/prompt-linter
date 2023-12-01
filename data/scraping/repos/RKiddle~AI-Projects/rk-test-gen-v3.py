from openai import OpenAI
from docx import Document

client = OpenAI()

completion = client.chat.completions.create(
  model="gpt-3.5-turbo",
  messages=[
    {"role": "system", "content": "You are a teacher"},
    {"role": "user", "content": "Write 5 questions on 9th grade elctricity each question has 4 multiple choice answers."}
  ]
)


generated_text = str(completion.choices[0].message)

# Save the generated text as a Word document
document = Document()
document.add_paragraph(generated_text)
document.save('generated_text.docx')
