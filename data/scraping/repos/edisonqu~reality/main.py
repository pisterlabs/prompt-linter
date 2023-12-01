from termcolor import colored
from whisper import record
from embeddings import *
from database import *

import openai
import dotenv
import os

dotenv.load_dotenv()
COHERE_KEY = os.getenv("COHERE_KEY")
from chromadb.utils.embedding_functions import CohereEmbeddingFunction

cohere_ef = CohereEmbeddingFunction(
    api_key=COHERE_KEY, model_name="large")

collection = database_initialization_and_collection(cohere_ef)
def print_cool_colors(option):
    if option == 1:
        print(colored("Hey, I am your AI assistant that can answer any questions you have about anything you've said today.", "blue"))
        # print(colored("Please give me a few seconds to get set up...", "blue"))
        # print()
        # print()
        # print(colored("READY!", "green"))
        question = input("Ask it on the command line here --> ")
        queries = int(input("How many results do you want? --> "))

        response = (query_db(collection=collection,query_text=question,embedding_function=cohere_ef,n_results=queries))
        # print(response)
        related_text = response.get('documents')[0]
        time = response.get('ids')[0]
        lowest_distance = response.get("distances")[0]

        for i in range(len(related_text)):
            print(colored(f"SEARCH RESPONSE #{i+1}",'blue'))
            print(colored(f"Found related text:",'green'))
            print(related_text[i],"with a distance score of ",lowest_distance[i], "and was generated on",time[i])

        print(colored("Running through GPT to give an answer, give me a moment ...", "blue"))

        import os
        import openai
        openai.api_key = os.getenv("OPENAI_API_KEY")

        completion = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "assistant", "content": f"You are a system that answers questions based on context provided to you and to provide an answer even if there is none. here is the quesiton: {question}, here is the context: {related_text[0]}"}
            ]
        )

        print(completion.choices[0].message.content)

    elif option == 2:
        print(colored("Please give me a few seconds to get set up...", "blue"))
        items = query_all(collection=collection).get('documents')
        print(colored("Here are all the conversations you heard today.", "green"))
        print()
        for item in items:
            print("--------------------------------------------------------------------------------------------------------------------------------")
            print(item)
        print( "--------------------------------------------------------------------------------------------------------------------------------")
    elif option == 3:
        print(colored("Option 3: Recording Started", "yellow"))
        print(colored("Please give me a few seconds to get set up...", "blue"))

        record(cohere_ef, collection)

    elif option == 4:
        print(colored("Please give me a few seconds to get set up...", "blue"))

        question = input("Ask it on the command line here --> ")
        queries = int(input("How many results do you want? --> "))

        response = (
            query_db(collection=collection, query_text=question, embedding_function=cohere_ef, n_results=queries))
        # print(response)
        related_text = response.get('documents')[0]
        time = response.get('ids')[0]
        lowest_distance = response.get("distances")[0]

        for i in range(len(related_text)):
            print(colored(f"SEARCH RESPONSE #{i + 1}", 'blue'))
            print(colored(f"Found related text:", 'green'))
            print(related_text[i], "with a distance score of ", lowest_distance[i], "and was generated on", time[i])
    else:
        print(colored("Invalid option, input numbers only from 1-3.", "red"))

def main():
    print("Welcome to your second brain! This program runs throughout, here are your current options")
    print()
    print(colored("Option 1: Ask questions to your brain database right now:", "red"))
    print()
    print(colored("Option 2: See your brain database", "green"))
    print()
    print(colored("Option 3: Start Recording", "yellow"))
    print()
    print(colored("Option 4: Search your brain database", "blue"))
    print()

    try:
        user_input = int(input("Enter your choice: "))
        print_cool_colors(user_input)
    except ValueError:
        print(colored("Invalid option. Please enter a number between 1 and 4.", "red"))

if __name__ == "__main__":
    main()
