import openai
import pyttsx3
import speech_recognition as sr
import random

# Set OpenAI API key
openai.api_key = "sk-CDUDYs3rDIq1n4L9iS2AG3B2bkFJzWEvKLw8ys9ARv19D3DO"
model_id = 'gpt-3.5-turbo'

# Initialize the text-to-speech engine
engine = pyttsx3.init()

# Change speech rate
engine.setProperty('rate', 180)

# Get the avaiable voice
voices = engine.getProperty('voices')

# Choose a voice based on the voice id
engine.setProperty('voice', voices[0].id) 

# Counter just for interacting purposes
interaction_counter = 0 


def transcribe_audio_to_text(filename):
    recognizer = sr.Recognizer()
    with sr.AudioFile(filename) as source:
        audio = recognizer.record(source)
        try:
            return recognizer.recognize_google(audio)
        except:
            print("")
            #print('Skipping unknown error')


def ChatGPT_conversation(conversation):
    response = openai.ChatCompletion.create(
        model=model_id,
        messages=conversation
    )
    api_usage = response['usage']
    print('Total token consumed: {0}'.format(api_usage['total_tokens']))
    conversation.append({'role': response.choices[0].message.role, 'content': response.choices[0].message.content})
    return conversation


def speak_text(text):
        engine.say(text)
        engine.runAndWait()


# Starting conversation
conversation = []
conversation.append({'role': 'user', 'content': 'chat with me as you be Friday AI from Iron Man, please make a one sentence phrase introducing yourself without saying something that sounds like this chat its already started'})
conversation = ChatGPT_conversation(conversation)
print('{0}: {1}\n'.format(conversation[-1]['role'].strip(), conversation[-1]['content'].strip()))
speak_text(conversation[-1]['content'].strip())


def activate_assistant():
    starting_chat_phrases = ["Yes sir, how may I assist you?",
               "Yes, What can I do for you?",
               "How can I help you, sir?",
               "Friday at your service, what do you need?",
               "Friday here, how can I help you today?",
               "Yes, what can I do for you today?",
               "Yes sir, what's on your mind?",
               "Friday ready to assist, what can I do for you?",
               "At your command, sir. How may I help you today?",
               "Yes, sir. How may I be of assistance to you right now?",
               "Yes boss, I'm here to help. What do you need from me?",
               "Yes, I'm listening. What can I do for you, sir?",
               "How can I assist you today, sir?",
               "Friday here, ready and eager to help. What can I do for you?",
               "Yes, sir. How can I make your day easier?",
               "Yes boss, what's the plan? How can I assist you today?",
               "Yes, I'm here and ready to assist. What's on your mind, sir?"]

    continued_chat_phrases = ["yes", "yes, sir", "yes, boss", "I'm all ears"]

    random_chat = ""
    if(interaction_counter == 1):
        random_chat = random.choice(starting_chat_phrases)
    else:
        random_chat = random.choice(continued_chat_phrases)
    
    return random_chat


def append_to_log(text):
    with open("chat_log.txt", "a") as f:
        f.write(text + "\n")


while True:

    #wait for users to say "Friday"
    print("Say 'Friday' to start...")
    recognizer = sr.Recognizer()

    with sr.Microphone() as source:
        audio = recognizer.listen(source)
        try:
            transcription = recognizer.recognize_google(audio)
            if "friday" in transcription.lower():
                interaction_counter += 1
                # Record audio
                filename = "input.wav"
                readyToWork = activate_assistant()
                speak_text(readyToWork)
                print(readyToWork)
                recognizer = sr.Recognizer()
                with sr.Microphone() as source:
                    source.pause_threshold = 1
                    audio = recognizer.listen(source, phrase_time_limit=None, timeout=None)
                    with open(filename, "wb") as f:
                        f.write(audio.get_wav_data())

                # Transcribe audio to text
                text = transcribe_audio_to_text(filename)
                if text:
                    print(f"You said: {text}")
                    append_to_log(f"You: {text}\n")

                    # Generate response using chatGPT
                    print(f"Friday says: {conversation}")

                    prompt = text
                    conversation.append({'role': 'user', 'content': prompt})
                    conversation = ChatGPT_conversation(conversation)

                    print('{0}: {1}\n'.format(conversation[-1]['role'].strip(), conversation[-1]['content'].strip()))

                    append_to_log(f"Friday: {conversation[-1]['content'].strip()}\n")

                    # Read response using text-to-speech
                    speak_text(conversation[-1]['content'].strip())

                    # In future maybe a conversation.clear to decrease input tokens as the conversation evolves ...
                        
        except Exception as e:
            continue
            #print("An error occurred: {}".format(e))