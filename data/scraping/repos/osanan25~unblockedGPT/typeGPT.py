from unblockedGPT.typeresponse import Typeinator
from unblockedGPT.auth import Database
import openai
import time

def typeGPT() -> None:
    """
        function that types the response of a prompt from GPT-3.5 or GPT-4
        input: None
        output: None
    """
    auth = Database.get_instance()
    while True:
        #print selected model
        model_selection = input("Selected model (1)gpt3.5 or (2)gpt.4\nEnter 1 or 2:")
        if model_selection != "1" and model_selection != "2":
            print("Invalid selection")
            continue
        
        prompt = input("Enter your prompt:")
        openai_api_key = auth.get_settings(0)
        openai.api_key = openai_api_key
        models = ['','gpt-3.5-turbo', 'gpt-4']
        try:
            response = openai.ChatCompletion.create(
                model=models[int(model_selection)],
                messages=[{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": prompt}]
            )
            chatbot_response = response['choices'][0]['message']['content'].strip()
        except:
            #print error message
            print("Run 'chat' command and update API key")
            continue
        if chatbot_response != "":
            print(chatbot_response)
            
            if input("Type this? (y/n)") == "y":
                print("Starting in 5 seconds...")
                time.sleep(5)
                typeinator = Typeinator()
                typeinator.type(chatbot_response)
            else:
                print("Not typing")


if __name__ == '__main__':
    typeGPT()
