import speech_recognition as sr
import openai
from text_to_speech import Speak



def transcribe_speech():
    '''
    This function uses the microphone to listen for 10 seconds and then returns the transcribed text. 
    '''

    # Create a recognizer object
    recognizer = sr.Recognizer()

    # Set the duration for listening
    duration = 10  # Number of seconds to listen

    # Use the default microphone as the audio source
    with sr.Microphone() as source:
        print("Listening...")
        # Adjust for ambient noise for better recognition
        recognizer.adjust_for_ambient_noise(source)

        # Record audio for the specified duration
        audio = recognizer.listen(source, timeout=duration)

    print("Finished recording.")

    try:
        # Recognize the speech using the default API
        transcript = recognizer.recognize_google(audio)
        return transcript
    except sr.UnknownValueError:
        return "Speech recognition could not understand audio"
    except sr.RequestError:
        return "Could not request results from the speech recognition service"


def ask_question(prompt):
    # Set up OpenAI API credentials
    openai.api_key = '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'

    # Set the GPT-3.5 model name
    model_name = "gpt-3.5-turbo"

    # Generate the question using GPT-3.5 chat model
    response = openai.ChatCompletion.create(
        model=model_name,
        messages=[
            {"role": "system", "content": "You are a helpful assistant in the year 2023."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=100,
        temperature=0.7
    )

    # Extract and return the generated answer
    answer = response['choices'][0]['message']['content'].strip()
    return answer

def main():

    patient_info = {}
    patient_info["questions"] = []
    intro = ("Greetings, this is a United States Army device, my purpose is to assess your "
         "injury and report back to my home base, and then proceed with further instructions. "
         "Do you have any questions or comments?")
    Speak(intro)


    # Get the transcribed text
    transcribed_text = transcribe_speech()
    print(transcribed_text)

    prompt = f"Check if '{transcribed_text}' is a question or statement. If it is a question, " \
         "return a tuple (1, question answered using the following background information " \
         "and any relevant information): " \
         "We are attempting to remotely assess your condition so that we can provide appropriate medical attention. " \
         "The test will consist of assessing your eye, verbal, and motor response. This device is part of a UAV system " \
         "sent by the United States Army. A medic will be here to assist you shortly. " \
         f"Otherwise, If it is not a question, return a tuple (0, '{transcribed_text}')"


    # Generate the question and get the answer
    answer = ask_question(prompt)
    Speak(answer)

    # Store relevant information in a dictionary
    if answer[0] == 0:
        patient_info["extra_info"] = answer[1]
    else:
        patient_info["questions"].append(transcribed_text)
        print(answer)
        Speak(answer[1])


    #Ask question
    question = "Can you tell me what year it is right now?"
    Speak(question)

    #Get patient response
    transcribed_text = transcribe_speech()
    print(transcribed_text)

    # Using a special delimiter "__RESPONSE__" to separate the context from the patient's response
    prompt = f"Patient was asked what year it was. It is currently 2023 but the patient responded with '{transcribed_text}'. Is the patient:\n\n" \
         "5: Oriented. A response with 2023 in it indicated they are oriented.\n " \
         "4: Confused conversation. Mentions the wrong year\n" \
         "3: Inappropriate words. No mention of years in the response.\n" \
         "2: Incomprehensible speech. Response can not be understood.\n" \
         "1: No response. Nothing is said. return a tuple (numerical score in integer form, answer)"

    # Generate the question and get the answer
    answer = ask_question(prompt)

    # Print and store the answer
    print("Answer:", answer)



    # Ask the question
    question = "Can you tell me who the current President of the United States is right now?"
    Speak(question)

    #Get patient response
    transcribed_text = transcribe_speech()
    print(transcribed_text)

    # Using a special delimiter "__RESPONSE__" to separate the context from the patient's response
    prompt = f"Patient was asked who the current President of the United States is. The right answer is Joe Biden but the patient responded with '{transcribed_text} ' Is the patient:\n\n" \
     "5: Oriented. A response with the correct President's name indicated they are oriented.\n " \
     "4: Confused conversation. Mentions the wrong President's name.\n" \
     "3: Inappropriate words. No mention of the President in the response.\n" \
     "2: Incomprehensible speech. Response cannot be understood.\n" \
     "1: No response. Nothing is said."

    # Generate the question and get the answer
    answer = ask_question(prompt)

    # Print and store the answer
    print("Answer:", answer)

    print("___________________________________")
    print("___________TEST COMPLETE___________")
    print(patient_info)

if __name__ == '__main__':
    main()
