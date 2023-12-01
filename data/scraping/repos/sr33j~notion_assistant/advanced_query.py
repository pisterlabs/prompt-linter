import pandas as pd
from openai import OpenAI
from embed import get_embedding
import numpy as np

client = OpenAI()

def generate_response(prompt):
    system_prompt = open("prompts/deduction_prompt.txt", "r").read()
    completion = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt}
    ]
    )
    return completion.choices[0].message.content  

def cosine_similarity(a, b):
    return np.dot(a, b)/(np.linalg.norm(a)*np.linalg.norm(b))

def get_question_breakdown(query):
    ## get prompt
    prompt = f"""
    For the question delimited by triple backticks, Can you give me exactly three prerequisite sub-questions that will help me answer the question? Please format your answer as a dash, subquestion, and newline.
    Example:
    ```What job am I best suited for?```
    - What am I particularly skilled at?
    - What am I intellectually curious about?
    - Does this provid value to the world? 

    Actual:
    """
    prompt += f"```\n{query}\n```"

    system_prompt = open("prompts/breakdown_prompt.txt", "r").read()

    ## generate response
    completion = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt}
    ])

    print("--- Breaking down " + query + " ---")
    # print(prompt)

    ## clean response
    all_questions_string = completion.choices[0].message.content  
    all_questions = all_questions_string.split("\n")
    subquestions = []
    for question in all_questions:
        if question.startswith("-"):
            subquestions.append(question[1:])
            print(subquestions[-1])
    print("------------------------")

    return subquestions


def get_docs_related_to_query(query, df, num_docs=4, cosine_threshold=0.1):
    query_embedding = get_embedding(query)
    ## get the embeddings from the csv file and calculate the cosine similarity
    df['cosine_similarity'] = df['Embedding'].apply(lambda x: cosine_similarity(x, query_embedding))

    ## sort the dataframe by cosine similarity
    df = df.sort_values(by=['cosine_similarity'], ascending=False)
    top_docs = df.head(num_docs)
    top_docs = top_docs[top_docs['cosine_similarity'] > cosine_threshold]

    ## get the prompt from the top docs
    return top_docs['Page Text'].tolist()

def get_prompt_from_docs(query, docs):
    prompt = "Keep your answer to less than 50 words."
    for doc in docs:
        prompt += "```\n"
        prompt += doc 
        prompt += "\n```\n"
    prompt += "Question: " + query + "\nAnswer:"
    return prompt

def use_reference_to_answer_subquestion(subquestion, docs_df):
    docs = get_docs_related_to_query(subquestion, docs_df)
    prompt = get_prompt_from_docs(subquestion, docs)

    ## generate response
    simple_prompt = open("prompts/simple_prompt.txt", "r").read()
    completion = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "system", "content": simple_prompt},
        {"role": "user", "content": prompt}
    ])

    print("--- Answering " + subquestion + " ---")
    answer = completion.choices[0].message.content
    print(answer)
    print("------------------------")

    return answer

def get_prompt_from_subanswers(query, subquestions, subanswers):
    prompt = ""
    for subquestion, subanswer in zip(subquestions, subanswers):
        prompt += "```\n"
        prompt += "Question: " + subquestion + "\n"
        prompt += "Answer: " + subanswer + "\n"
        prompt += "```\n"
    prompt += "Based on the answers to these questions, what is the answer to " + query + "?"
    prompt += " Keep your answer to less than 50 words. Please be specific and concrete in your answer."
    return prompt

def generate_response_from_subanswers(query, subquestions, subanswers):
    ## get prompt
    prompt = get_prompt_from_subanswers(query, subquestions, subanswers)

    ## generate response
    system_prompt = open("prompts/synthesis_prompt.txt", "r").read()
    completion = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt}
    ])

    print("--- SYNTHESIS ---")
    answer = completion.choices[0].message.content
    print(answer)
    print("------------------------")

    return answer

def main():
    ## read in the csv file
    df = pd.read_csv('notion_embeddings.csv')
    df['Embedding'] = df['Embedding'].apply(lambda x: np.fromstring(x, sep=','))
    ## get input from the user for a query
    while True:
        query = input("Ask a question about yourself: ")
        
        print("===== BREAKING DOWN ORIGINAL QUESTION =====")
        
        subquestions = get_question_breakdown(query)

        print("===== ANSWERING SUBQUESTIONS WITH REFERENCES =====")
        subanswers = []
        for subquestion in subquestions:
            subanswer = use_reference_to_answer_subquestion(subquestion, df)
            subanswers.append(subanswer)

        print("===== ANSWERING ORIGINAL QUESTION BASED ON DEDUCTIONS =====")
        answer = generate_response_from_subanswers(query, subquestions, subanswers)
        # print(answer)

if __name__ == "__main__":
    main()