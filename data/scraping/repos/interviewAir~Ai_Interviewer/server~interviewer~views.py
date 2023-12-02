from rest_framework.decorators import api_view
from rest_framework.response import Response
from django.http import JsonResponse
import openai
import speech_recognition as sr
from django.conf import settings
from gtts import gTTS

conversation_context = {}
openai.api_key = settings.API_KEY


def get_context(request):
    if request.method == "POST":
        position = request.data.get("position")
        level = request.data.get("level")
        interview_type = request.data.get("type")

        prompt = f"Interview a candidate for the position of a {position} at the {level} level.\
                You are at the highest level of that position.\
                The interview should be conducted in a professional manner.\
                It should focus on the {interview_type} aspects of the\
                job."

        conversation_context["conversation_history"].append({"role": "AI", "content": prompt})

        response = openai.Completion.create(
            engine="text-davinci-003",
            prompt=prompt,
            max_tokens=100,
        )

        conversation_context["conversation_history"].append({"role": "AI", "content": response.choices[0].text})

        return response.choices[0].text
    if request.method == "POST":
        recognizer = sr.Recognizer()

        with sr.Microphone() as source:
            try:
                recognizer.adjust_for_ambient_noise(source)
                audio = recognizer.listen(source)
                transcription = recognizer.recognize_google(audio)

                openai.api_key = settings.API_KEY
                response = openai.Completion.create(
                    engine="text-davinci-003",
                    prompt=f"Rate my answer to the {question} on a scale of one to 10",
                    max_tokens=100,
                )
                print(response.choices[0].text)

            except sr.UnknownValueError:
                transcription = "Inaudible..."

            except sr.RequestError as e:
                transcription = "Could not request results from Google Speech Recognition service; {0}".format(
                    e
                )
            except KeyboardInterrupt:
                pass

            except Exception as e:
                print(e)
    else:
        transcription = "Press the button and start speaking"
