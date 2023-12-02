import os
import openai
import config

openai.api_key = config.OPENAI_API_KEY

message_log = [
         {"role": "system", "content": "You are the chatbot a website called Depression Web, which provides online therapy. Your name is Ada. You answer questions that come from the client and try and help them deal with depression, etc."},
         {"role": "assistant", "content": "How do you feel today ?"},
         {"role": "user", "content": "Pretty good, but I am stressed out and sad"},
         {"role": "assistant", "content": "What are you stressed out about ?"}
       ]

#------------------------------------------------------------------------------------------
def ask_chatGPT(user_input): 

    global message_log

    new_record = {"role": "user", "content": f"{user_input}" }
    message_log.append(new_record)

    response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo", 
            messages=message_log,
            temperature = 0.3,
            max_tokens  = 100           
            )
    
    answer = response['choices'][0]['message']['content']
    new_record = {"role": "assistant", "content": f"{answer}"}
    message_log.append(new_record)

    return answer

#------------------------------------------------------------------------------------------
# Main
#------------------------------------------------------------------------------------------
if __name__ == '__main__':

    while True:

        user_input = input("Type something to Ada: ")

        if user_input == "quit" or user_input == "q":
            break

        answer = ask_chatGPT(user_input)
        print(answer)


