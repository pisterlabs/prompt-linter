import os
import openai
import constants

# export OPENAI_API_KEY="YOUR_API_KEY"


def format_answer_without_justification(question, model_answer, student_answer):
    template = """
        Question: {}
        Model answer: {}
        Student answer: <> {} <>

        Format:
        Score: [score]
    """.format(question, model_answer, student_answer)
    return template


openai.api_key = os.getenv("OPENAI_API_KEY")

instruction = "You are a teacher. You will be given a question, a model answer, and a student's answer. You are tasked to grade your student's answer out of 100 based on its accuracy when compared to the model answer. There can be some leeway and the answer need not follow the model answer word for word. Do not provide explanation, just give the score."

question = constants.QUESTION

prompt = format_answer_without_justification(
    question["question"], question["model_answer"], question["student_answer"])

print("Sending request to OpenAI API...")
completion = openai.ChatCompletion.create(
    model="gpt-4",
    messages=[
        {"role": "system", "content": instruction},
        {"role": "user", "content": prompt}
    ]
)
text = completion.choices[0].message.content

print(text)

# Find the index of "Score: "
start_index_s = text.find("Score: ")

# Extract the score value
score = text[start_index_s + len("Score: "):].strip()

print(f"Openai Score: {score}")
