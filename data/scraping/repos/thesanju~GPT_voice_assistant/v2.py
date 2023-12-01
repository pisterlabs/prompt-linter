import openai
import speech_recognition as sr
import pyttsx3

engine = pyttsx3.init()
listener = sr.Recognizer()

openai.api_key = "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"

# Define the keyword to stop listening
stop_keyword = "stop listening"

while True:
    with sr.Microphone() as source:
        print("Speak now...")
        voice = listener.listen(source)
        data = listener.recognize_google(voice)
        model = "text-davinci-003"

        if stop_keyword.lower() in data.lower():
            print("Listening stopped.")
            break

        completion = openai.Completion.create(
            model="text-davinci-003",
            prompt=data,
            max_tokens=1024,
            temperature=0.5,
            n=1,
            stop=None
        )
        response = completion.choices[0].text
        choice = int(input("Press 1 to print the response or press 2 to print and hear the response: "))

        if choice == 1:
            print(response)
        else:
            print(response)
            engine.say(response)
            engine.runAndWait()

    repeat = input("Do you want to ask more questions?: ")
    if repeat.lower() in ["no", "n"]:
        break
