import tiktoken
from openai import OpenAI
openai = OpenAI()

def get_num_tokens(string):
    embedding_encoding = "cl100k_base"  # this the encoding for text-embedding-ada-002
    encoding = tiktoken.get_encoding(embedding_encoding)
    return len(encoding.encode(string))

def get_embedding(input: str):
    response = openai.embeddings.create(
      input=input,
      model="text-embedding-ada-002"
    )
    return response.data[0].embedding

def clean_answer(answer: str):
    return answer.replace("\n", " ")

def get_answers_string(answers: list):
    answers_str = '\n'.join(map(lambda x: '- "' + clean_answer(x) + '"', answers))
    return answers_str


def summarise_answers(survey_context: str, question: str, answers: list):
    completion = openai.chat.completions.create(
        model="gpt-4-1106-preview",
        messages=[
          # {"role": "system", "content": "You are a helpful assistant."},
          {"role": "user", "content": f'I am analysing results from a survey. {survey_context}\n\nHere are some answers to the survey question "{question}"\n\nAnswers:\n{get_answers_string(answers)}\n\nWhat is the common theme in these answers? Respond directly with the common theme, without any preamble.'}
        ],
        temperature=0.1,
        max_tokens=100
    )
    return completion.choices[0].message.content


def get_alphabetical_list(len: int):
    return list(map(lambda x: chr(x + 97).capitalize(), list(range(0, len))))

def get_classification_prompt(question: str, answer: str, options: list):
    labels = get_alphabetical_list(len(options))
    labelled_options = "\n".join([f"{letter}. {option}" for letter, option in zip(labels, options)])

    prompt = '\n\n'.join([
        f"Here is a survey question, and an answer from one of the participants:",
        f"Q: {question}\nA: {clean_answer(answer)}",
        f"Please classify the answer as one of the following options:",
        labelled_options,
        f"Respond directly with the label and nothing else e.g. \"{labels[0]}\""
    ])

    return prompt, labels


def classify_answer(question: str, answer: str, options: list):
    
    prompt, labels = get_classification_prompt(question, answer, options)
    
    completion = openai.chat.completions.create(
        model="gpt-4-1106-preview",
        messages=[
          # {"role": "system", "content": "You are a helpful assistant."},
          {"role": "user", "content": prompt}
        ],
        temperature=0,
        max_tokens=1
    )
    label = completion.choices[0].message.content[0]

    try:
        return labels.index(label)
    except ValueError:
        print(f"Label {label} not in list of labels {labels}")
        return None
    
    