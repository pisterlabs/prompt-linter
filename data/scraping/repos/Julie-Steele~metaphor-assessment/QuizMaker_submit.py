#hello! this is a quiz maker that takes in a topic and makes an interactive quiz on it based on internet pages. 

#takes in topic, finds relevant websites
#takes in websites, makes multiple choice questions with right answers
#takes questions, makes interactive quiz
#if it isn't working, let me know @jssteele@mit.edu

import json
import os
import openai
from metaphor_python import Metaphor


#GENERAL 
openai.api_key = os.getenv("OPENAI_API_KEY")
metaphor = Metaphor(os.getenv("METAPHOR_API_KEY"))

print("What do you want to quiz yourself on?")

USER_QUESTION = input()#"Famous Art Styles"


#makes internew query from prompt 
SYSTEM_MESSAGE = "You are a helpful assistant that generates search queiries based on user questions. Only generate one search query. Keep it general. Facts about <query> or something like that."

completion = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "system", "content": SYSTEM_MESSAGE},
        {"role": "user", "content": USER_QUESTION},
    ],
    
)


query = completion.choices[0].message.content

#print(f"Query: {query}\n")

search_response = metaphor.search(
    query, use_autoprompt=True, start_published_date="2023-06-01"
)
#print(f"URLs: {[result.url for result in search_response.results]}\n")



#makes questions from webcontent. gpt replies with json with question, answers, and correct answer
SYSTEM_MESSAGE = "You are a helpful assistant that turns the webpage into a 4 part multiple choice quiz question. Return the question, the answer options 1 2, 3, 4, and the correct number. Absolitely no A-D. You must label with numbers. Never make the question reference the article. Make it possible to answer without reading the article."
def get_completion(result):
    return openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": SYSTEM_MESSAGE},
            {"role": "user", "content": result.extract},
        ],
        functions = [
    {
        "name": "make_quiz",
        "description": "takes in a quiz question, 4 answer choices (1, 2, 3, 4), and the correct number, and turns it into a quiz",
        "parameters": {
        "type": "object", 
        "properties": {
            "quiz_question": {
            "type": "string", 
            "description": "question about the topic"
            },
            "answer_choices": {
            "type": "string", 
            "description": "four answer choices labeled 1, 2, 3, 4"
            },
            "correct": {
            "type": "integer", 
            "description": "the correct answer choice"
            },
        },
        #"required": ["properties", "answer_choices", "correct"]
        }
    }
    ],
            function_call={"name": "make_quiz"}
    )

contents_result = search_response.get_contents()

num_questions = 3


questions = []
answers = []
correct = []

print(num_questions, "question quiz on", USER_QUESTION, "Type 1, 2, 3, or 4 then press enter")
print("")

#takes websites and calls the chatgpt function above to make questions
for i in range(num_questions):
    result = contents_result.contents[i]
    try:
        summary = get_completion(result).choices[0].message
        arguments = json.loads(summary["function_call"]["arguments"])
        questions.append(arguments["quiz_question"])
        answers.append(arguments["answer_choices"])
        correct.append(arguments["correct"])
    except:
        print("Sorry, error with question", i+1, "skipping")


#interactive quiz game 
score = 0

for i in range(num_questions):
    print(questions[i])
    print(answers[i])
    choice = input()
    try:
        choice = int(choice)
    except:
        print("Invalid input. Type 1, 2, 3, or 4 then press enter. try again")
        choice = input()
    if int(choice) == correct[i]:
        score += 1
        print("Correct!")
    else:
        print("Incorrect, answer is", correct[i])
    print("")


print("Score", score, "out of", num_questions, "on the", USER_QUESTION, "quiz")