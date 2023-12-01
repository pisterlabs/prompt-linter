import openai
import csv
import env
import os

env.addEnv()
openai.api_key = os.environ['OPEN_AI_KEY']

# Read the CSV file and extract the necessary data
data = []
with open('data/SavedData/Year-wise Telecom Subscribers from 2008 to 2022.csv', 'r') as file:
    reader = csv.reader(file)
    header = next(reader)  # Assuming the first row contains column names
    for row in reader:
        data.append(row)

# Prepare the context using the data from the CSV
context = ' '.join([' '.join(row) for row in data])
# max_context_length = 4096  # Maximum allowed context length
# context = context[:max_context_length]
# Generate the answer using the OpenAI API


# Make an API request
response = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "total Wireless subscription in Mumbai"},
        {"role": "assistant", "content": context}
    ]
)

# Extract and display the answer
answer = response['choices'][0]['message']['content']
print("Answer:", answer)

# def generate_answer(question):
#     prompt = f"""Use the below context csv data to answer the subsequent question. If the answer cannot be found, write "I don't know."
#             Context:
#             \"\"\"
#             {context}
#             \"\"\"
#             Question: {question}?"""
#     print(prompt)
#     response = openai.Completion.create(
#         engine='davinci',
#         prompt=prompt,
#         max_tokens=100,
#         n=1,
#         stop=None,
#         temperature=0.5
#     )
#     answer = response.choices[0].text.strip()
#     return answer


# # Ask a question and get the answer
# question = 'total Wireless subscription in Mumbai'
# answer = generate_answer(question)
# print(answer)
