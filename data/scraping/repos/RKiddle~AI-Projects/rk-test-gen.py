from openai import OpenAI


client = OpenAI()

completion = client.chat.completions.create(
  model="gpt-3.5-turbo",
  messages=[
    {"role": "system", "content": "You are a teacher"},
    {"role": "user", "content": "Write 5 questions on 9th grade elctricity each question has 4 multiple choice answers."}
  ]
)


generated_message = str(completion.choices[0].message)

# Save the text in a file
with open("generated_text.txt", "w") as file:
    file.write(generated_message.strip())

print("The Text Has Been Generated Successfully!")