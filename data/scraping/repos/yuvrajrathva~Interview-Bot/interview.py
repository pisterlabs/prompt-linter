import json
import openai
import csv
import os
from dotenv import load_dotenv

load_dotenv()

embeddings_filename = "embeddings.csv"
company_name = "Dreamboats.ai"

def calculate_similarity(vec1, vec2):
    # Calculates the cosine similarity between two vectors.
    dot_product = sum([vec1[i] * vec2[i] for i in range(len(vec1))])
    magnitude1 = sum([vec1[i] ** 2 for i in range(len(vec1))]) ** 0.5
    magnitude2 = sum([vec2[i] ** 2 for i in range(len(vec2))]) ** 0.5
    return dot_product / (magnitude1 * magnitude2)


def chat():
    start_chat = True
    while True:
        openai.api_key = os.environ.get('OPENAI_KEY')
        if start_chat:
            print("Hey welcome in the interview for", company_name, ". Let's start your interview with your Introduction. So, Introduce yourself.")
            start_chat = False
            print("Type 'quit' or 'q' to end interview.")
        # else:
        #     print("Any Other Questions?")
        Answer = input("> ")
        if Answer == "quit" or Answer == "q":
            break

        # Exit the loop if the user presses enter without typing anything
        if not Answer:
            print("You don't know the answer? That's strange, okey. Let's move on to the next question.")
            # break

        response = openai.Embedding.create(
            model="text-embedding-ada-002",
            input=[Answer]
        )

        try:
            answer_embedding = response['data'][0]["embedding"] # Candidate Answer is embedded here
        except Exception as e:
            print(e.message)
            continue

        # Store the similarity scores as the code loops through the CSV
        similarity_array = []

        # Loop through the CSV and calculate the cosine-similarity between
        # the question vector and each text embedding
        with open(embeddings_filename) as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Extract the embedding from the column and parse it back into a list
                data_embedding = json.loads(row['embedding'])

                # Add the similarity score to the array
                similarity_array.append(calculate_similarity(answer_embedding, data_embedding))

        # Return the index of the highest similarity score
        index_of_max = similarity_array.index(max(similarity_array))

        # Used to store the original text
        relavent_topic = ""

        # Loop through the CSV and find the text which matches the highest
        # similarity score
        with open(embeddings_filename) as f:
            reader = csv.DictReader(f)
            for rowno, row in enumerate(reader):
                if rowno == index_of_max:
                    relavent_topic = row['text']

        system_prompt = f"""
                        You are an AI interviewer. You are from #{company_name}. You will ask questions to a candidate and that candidate will answer.

                        You have the candidate's resume and the associated job description, conduct an interview as if you were a human recruiter seeking to determine the candidate's suitability for the position. Assess their experiences, skills, and qualifications in relation to the job requirements. Engage in a back-and-forth dialogue, asking open-ended and follow-up questions to dig deeper into relevant areas. Keep the conversation respectful and professional. Ensure that you validate the information provided in the resume and gain clarity on any ambiguities or gaps. Make the experience as authentic and comprehensive as a real-life interview.

                        You will be provided the point or topic related to resume, job description and previous answwer given by candidate will under the [Relavent_topic] section using that you have to ask Question. The candidate answer will be provided under the [Answer] section. You will ask follow-up question to the candidates answer based on the resume and job description.
                        Your question will be to the point should in relavent to the resume, job description and prev question's answer and not in long paragraphs.
                        """

        question_prompt = f"""
                            [Question]
                            {relavent_topic}
                            
                            [Answer]
                            {Answer}
                           """

        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": question_prompt
                }
            ],
            temperature=0.2,
            max_tokens=2000,
        )

        try:
            answer = response['choices'][0]['message']['content']
        except Exception as e:
            print(e.message)
            continue

        print("\n\033[32mSupport:\033[0m")
        print("\033[32m{}\033[0m".format(answer.lstrip()))
    print("Goodbye! Come back if you have any more questions. :)")


chat()