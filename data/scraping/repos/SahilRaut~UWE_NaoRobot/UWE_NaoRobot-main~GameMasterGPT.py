###########################################################################################################################
## This Program instagates a 'chose your own adventure' game with GPT 3.5
###########################################################################################################################
## Author: Benjamin Mallinson
## Date Created: 25/10/23
## Version: Python 3.10
##################################################

import openai

openai.api_key = "
"

# Dictionary to store current message from user
message = {"role":"user", "content": input("Tell me where you would like to go today \nYou:")}
# List to store messages including first inistalisation system message
conversation = [{"role": "system", "content": "You are the story and question master of a choose your own adventure, the user is the charecter in a world they chose."}]

while(message["content"]):
    
    """ Main loop to itterate each user interaction """

    conversation.append(message) # Add last message to conversation list

    completion = openai.ChatCompletion.create( # Do Inference of GPT
        model="gpt-3.5-turbo",
        messages=conversation, # Send conversation
        temperature=1.3
    ) 

    conversation.append(completion.choices[0].message) # Add response to conversation for next loop
    message["content"] = input(f"Story Master: {completion.choices[0].message.content} \nYou:\n") # Update message content with a new response input
