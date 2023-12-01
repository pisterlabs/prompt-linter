# this is just a playground for me to test code on the console

import db
import openai


def get_and_print_user_input():
    # get user input
    user_input = input("Enter a number: ")
    # print user input
    print("You entered: " + user_input)


def call_chat_gpt_api():
    # call chatGPT API
    print("Calling chatGPT API")


def create_chat_gpt_request(prompt):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "system",
                "content": """You are a zen master who is enlightened 
            and knows everything about birds, including ancient secrets. You speak cryptically and,
            although you don't blatantly lie, you use your mystical language to make things
            seem as interesting as possible.""",
            },
            {"role": "user", "content": prompt},
            # {"role": "assistant", "content": "The Los Angeles Dodgers won the World Series in 2020."},
            # {"role": "user", "content": "Where was it played?"}
        ],
    )
    # print out the response, but formatted nicely.
    print(response)


# print("Input your prompt: ")
create_chat_gpt_request("tell me something amazing about parakeets")

# db_manager = db.DatabaseManager()
# print("updating document")
# temp = db_manager.update_document({"name": "John"}, {"$set": {"name": "John"}}, True)
# print("updated result")
# print(temp.raw_result)
# print("finding document")
# found_doc = db_manager.find_document({"name": "John"})
# print(found_doc)


# Path: app.py
