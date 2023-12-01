import os
import openai
import sys
from sr import listen, UnknownValueError, WaitTimeoutError
from tts import speak

# Load your API key from an environment variable or secret management service
openai.api_key = os.getenv("OPENAI_API_KEY")

def run():
    print("Listening...")
    query = listen()
    response = openai.Completion.create(model="text-davinci-003", prompt="You are a general intelligence AI. Respond to this query: " + query, temperature=0, max_tokens=2000)
    print(response.choices[0].text)
    speak(response.choices[0].text)

if __name__ == "__main__":
    print("\nWelcome to the OpenAI assistant! (press Ctrl+C to exit)")
    while True:
        try:
            # Wait for return to be pressed
            print("\n---------------\nPress return to start the assistant")
            input()

            # Run the assistant
            run()

        except KeyboardInterrupt:
            print("\nLater nerd!")
            sys.exit(0)
        
        except UnknownValueError:
            print("I didn't catch that. What did you say?\n")
            speak("I didn't catch that. What did you say?")

        except WaitTimeoutError:
            print("It doesn't seem like you said anything. Try again!\n")
            speak("It doesn't seem like you said anything. Try again!")
