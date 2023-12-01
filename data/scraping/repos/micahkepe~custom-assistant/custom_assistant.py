import speech_recognition as sr
import openai

from elevenlabslib import *

# Enter keys for OpenAI and ElevenLabs, then put voice model's name and your name
openai.api_key = 'OPEN AI KEY'

elevenLabsAPIKey = 'ELEVENLABS KEY'

model = "VOICE MODEL NAME"
your_name = "ENTER NAME"

r = sr.Recognizer()
mic = sr.Microphone()
user = ElevenLabsUser(elevenLabsAPIKey)

voice = user.get_voices_by_name(model)[0]

conversation = [
        {"role": "system", "content": f"Your name is {model} and you're an assistant for {your_name}."},
    ]

while True:
    with mic as source:
        r.adjust_for_ambient_noise(source)  # Can set the duration with duration keyword

        # Uncomment if you want model to greet you
        # print(f"{model}: How may I help you?")
        # voice.generate_and_play_audio("How may I help you?", playInBackground=False)

        print("Speak now...")

        try:
            # Gather audio and transcribe to text
            audio = r.listen(source)
            word = r.recognize_google(audio)

            # Print user's query
            print(f"You said: {word}")

            # Quit program when user says "That is all"
            if word.lower() == "that is all":
                print(f"{model}: See you later!")
                quit()

            # if user wants to assistant to draw something
            if "draw" in word:
                i = word.find("draw")
                i += 5
                response = openai.Image.create(
                    prompt=word[i:],
                    n=1,
                    size="1024x1024"
                )
                image_url = response['data'][0]['url']
                print(f"{model}: Here's {word[i:]}")
                print(image_url)
                print("=====")

            # if user asks a question
            else:
                conversation.append({"role": "user", "content": word})

                response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=conversation,
                max_tokens=4000,
                n=1,
                stop=None,
                temperature=0.5,
                )

                message = response["choices"][0]["message"]["content"]
                conversation.append({"role": "assistant", "content": message})
                # Print GPT's response
                print(f"{model}: {message}")

                # Have model speak generated response
                voice.generate_and_play_audio(message, playInBackground=False)
                print("=====")

        except Exception as e:
            print("Couldn't interpret audio, try again.".format(e))
            print("=====")
