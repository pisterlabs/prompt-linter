import openai
import speech_recognition as sr
import pyttsx3
import json
import time

recognizer = sr.Recognizer()
global model_id

with open('config/config.json') as f:
    config = json.load(f)
    openai.api_key = config['api_key']  # insert your API Key here
    language = config['language']       # Format: en-EN, en-US...
    model_id = config['model']

def chat_gpt_conversation(conversation):
    try:
        response = openai.ChatCompletion.create(
            model=model_id,
            messages=conversation,
        )
        api_usage = response['usage']
        print('Total token consumed: {0}'.format(api_usage['total_tokens']))
        # stop means complete
        print(response['choices'][0].finish_reason)
        # print(response['choices'][0].index)
        conversation.append({'role': response.choices[0].message.role, 'content': response.choices[0].message.content})
    except openai.error.RateLimitError as e:
        # Retry after waiting for a short period
        print("Rate limit exceeded. Retrying after 2 seconds...")
        time.sleep(2)
        return chat_gpt_conversation(conversation)
    return conversation

def text_to_speech(text,language):
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()

def main():
    conversation = [{'role': 'system', 'content': 'How may I help you?'}]
    operation = None
    while operation != "x" and operation != "X":
        print("""Please choose your preferred input method.
1: Use your own voice and convert speech to text
2: Input text directly from the keyboard
X: Exit""")
        operation = input("=> ")
        prompt = None
        if(operation == "1"):
            while True:
                with sr.Microphone() as source:
                    print("Say something...")
                    audio = recognizer.listen(source)

                try:
                    prompt = recognizer.recognize_google(audio,language=language)
                    if(prompt == 'x' or prompt == "X"):
                        break
                    print("You:", prompt)
                    conversation.append({'role': 'user', 'content': prompt})
                    
                except sr.UnknownValueError:
                    print("Google Speech Recognition could not understand audio")
                except sr.RequestError as e:
                    print("Could not request results from Google Speech Recognition service; {0}".format(e))

                conversation = chat_gpt_conversation(conversation)
                print('{0}: {1}\n'.format(conversation[-1]['role'].strip().capitalize(), conversation[-1]['content'].strip()))
                text_to_speech(conversation[-1]['content'].strip(),language)

        elif(operation == "2"):
            while True:
                prompt = input("Enter your prompt: ")
                if(prompt == 'x' or prompt == "X"):
                    break
                conversation.append({'role': 'user', 'content': prompt})
                conversation = chat_gpt_conversation(conversation)
                print('{0}: {1}\n'.format(conversation[-1]['role'].strip().capitalize(), conversation[-1]['content'].strip()))
                text_to_speech(conversation[-1]['content'].strip(),language)
        
if __name__ == "__main__":
    main()