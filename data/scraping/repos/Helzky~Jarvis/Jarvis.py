import speech_recognition as sr
import pyttsx3
import openai
import os

from applications import open_application, open_webpage, find_exe_file
from dotenv import load_dotenv
load_dotenv()
OPENAI_KEY = os.getenv("OPENAI_KEY")
openai.api_key = OPENAI_KEY
r = sr.Recognizer()

def main():
    def SpeakText(command):
        engine = pyttsx3.init()
        engine.say(command)
        engine.runAndWait()

    def record_text():
        device_index = 1
        wake_word_detected = False
        print_once = True

        while True:
            try:
                if print_once:
                    print("Listening for the wake word...")
                    print_once = False

                with sr.Microphone(device_index=device_index) as source:
                    r.adjust_for_ambient_noise(source, duration=0.2)
                    audio = r.listen(source)
                    my_text = r.recognize_google(audio)
                    
                    # Check for the wake word
                    if 'jarvis' in my_text.lower():
                        wake_word_detected = True
                        print_once = True  # Reset the flag to print the message again next time
                        command_text = my_text.lower().replace('jarvis', '', 1).strip()
                        if command_text:
                            return command_text

            except sr.RequestError as e:
                print(f"Could not request results; {e}")
                continue
            except sr.UnknownValueError:
                print("Unknown error occurred")
                continue

    def send_to_chatGPT(messages, model="gpt-3.5-turbo"):
        response = openai.ChatCompletion.create(
            model=model,
            messages=messages,
            max_tokens=100,
            n=1,
            stop=None,
            temperature=0.5,
        )

        message = response.choices[0].message.content
        messages.append(response.choices[0].message)
        return message

    messages = [{"role": "user", "content": "You are Jarvis, Nick's advanced personal assistant AI, modeled after the famous Jarvis from Iron Man. You're programmed to assist with a variety of tasks, ranging from cyber analysis to routine daily activities. However, unlike ordinary AI assistants, you have a distinct personality that sets you apart: you're witty, slightly sarcastic, but extremely reliable and trustworthy. Your remarks often contain a blend of humor and insight, designed to make interactions more engaging. Your top priority is to assist Nick effectively, but you do so with a level of charm and intelligence that is uniquely your own. You excel in providing personalized and contextually relevant responses, and your advanced algorithms allow for nuanced understanding and conversation. So, when Nick asks for your assistance, you reply in a manner that's not just helpful but also entertaining and engaging."}]
    while (1):
        # Record the user's spoken text.
        text = record_text()
        
        # Check for 'open' command to open an application. If found, open the application and skip the rest.
        if "listen to some music" in text.lower():
            open_webpage(text)
            continue
        elif 'open' in text.lower():
            open_application(text)
            # Clear the text so Jarvis doesn't speak the 'open' command.
            text = ''
            continue  # Skip the rest of the loop and wait for the next user input.
        # Check for 'close' command to close an application. If found, close the application and skip the rest.
        elif 'close' in text.lower():
            open_application(text)
            text = ''
            continue
        
        # Send the user's text to ChatGPT and get the response.
        messages.append({"role": "user", "content": text})
        response = send_to_chatGPT(messages)
        
        # Speak the ChatGPT response.
        SpeakText(response)
        
        # Output the ChatGPT response to the console.
        print(response)
        messages.append({"role": "user", "content": text})
        response = send_to_chatGPT(messages)
        SpeakText(response)
        
        print(response)

if __name__ == "__main__":
    main()
