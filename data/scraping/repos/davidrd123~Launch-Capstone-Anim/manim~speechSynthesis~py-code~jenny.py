import azure.cognitiveservices.speech as speechsdk
import os
from dotenv import load_dotenv
import openai
import inquirer

speech_key = os.environ.get('SPEECH_KEY')
service_region = os.environ.get('SPEECH_REGION')
openai_key = os.environ.get('OPENAI_KEY')

speech_config = speechsdk.SpeechConfig(subscription=speech_key, region=service_region)
speech_config.speech_synthesis_voice_name = "en-US-TonyNeural"
speech_config.speech_recognition_language="en-US"

audio_config = speechsdk.audio.AudioConfig(use_default_microphone=True)

MODEL = "gpt-3.5-turbo-16k"

def recognize_from_microphone(speech_recognizer):
    print("Speak into your microphone.")
    speech_recognition_result = speech_recognizer.recognize_once_async().get()

    if speech_recognition_result.reason == speechsdk.ResultReason.RecognizedSpeech:
        print("Recognized: {}".format(speech_recognition_result.text))
    elif speech_recognition_result.reason == speechsdk.ResultReason.NoMatch:
        print("No speech could be recognized: {}".format(speech_recognition_result.no_match_details))
    elif speech_recognition_result.reason == speechsdk.ResultReason.Canceled:
        cancellation_details = speech_recognition_result.cancellation_details
        print("Speech Recognition canceled: {}".format(cancellation_details.reason))
        if cancellation_details.reason == speechsdk.CancellationReason.Error:
            print("Error details: {}".format(cancellation_details.error_details))
            print("Did you set the speech resource key and region values?")

    return speech_recognition_result.text

def chat_gpt(message, conversation):
    conversation.append({"role": "user", "content": message})
    response = openai.ChatCompletion.create(
        model=MODEL,
        messages=conversation,
        temperature=0.5,
    )
    assistant_message = response["choices"][0]["message"]["content"]
    conversation.append({"role": "assistant", "content": assistant_message})
    return assistant_message, conversation

def create_ssml(text, speech_config=speech_config):
    ssml_template = """
    <speak version="1.0" xmlns="http://www.w3.org/2001/10/synthesis" xmlns:mstts="https://www.w3.org/2001/mstts" xml:lang="en-US">
        <voice name="{voice_name}">
            <prosody rate="fast">
                {content}
            </prosody>
        </voice>
    </speak>
    """
    return ssml_template.format(content=text, voice_name=speech_config.speech_synthesis_voice_name)

def speak_text(speech_synthesizer, text, speech_config=speech_config):
    ssml = create_ssml(text)
    # result = speech_synthesizer.speak_text_async(text).get()
    result = speech_synthesizer.speak_ssml_async(ssml).get()
    if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
        print("Speech synthesized for text [{}]".format(text))
    elif result.reason == speechsdk.ResultReason.Canceled:
        cancellation_details = result.cancellation_details
        print("Speech synthesis canceled: {}".format(cancellation_details.reason))
        if cancellation_details.reason == speechsdk.CancellationReason.Error:
            print("Error details: {}".format(cancellation_details.error_details))

def select_speech_or_text_input():
    options = ['Speech', 'Text']

    questions = [
        inquirer.List('input_type',
                    message="How would you like to input your message?",
                    choices=options,
                ),  
    ]
    try:
        answers = inquirer.prompt(questions)
        if answers is not None:
            return answers['input_type']
        else:
            print("No input selected. Defaulting to 'Text'.")
            return 'Text'
    except Exception as e:
        print(f"An error occurred: {e}. Defaulting to 'Text'.")
        return 'Text'


def chat_loop():
    import datetime
    import os

    speech_synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config)
    speech_recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config, audio_config=audio_config)

    message = """"""

    # Create a unique timestamped filename for the chat log
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    chat_log_filename = f"chat_logs/chat_log_{timestamp}.txt"

    # Ensure the chat_logs directory exists
    os.makedirs(os.path.dirname(chat_log_filename), exist_ok=True)

    conversation = [
        {"role": "system", "content": "You are a sassy and resentful assistant who still takes delight in being insightful and helpful despite himself - he loves brainstorming technical ideas and going deep into system design"},
    ]

    while True:
        response_text, conversation = chat_gpt(message, conversation)

        # Append the user message and assistant response to the chat log
        with open(chat_log_filename, 'a') as chat_log_file:
            chat_log_file.write(f"User: {message}\n")
            chat_log_file.write(f"Assistant: {response_text}\n")

        print(response_text)
        speak_text(speech_synthesizer, response_text)
        speech_or_text_input = select_speech_or_text_input()
        if speech_or_text_input == 'Speech':
            message = recognize_from_microphone(speech_recognizer)
        else:
            message = input("User: ")


if __name__ == "__main__":
    chat_loop()

# Example usage
# messages = [
#     "What is the capital of France?"
# ]


# response_text = chat_gpt(messages[0])


# text = "Hi, this is Jenny"

# # use the default speaker as audio output.
# speech_synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config)

# result = speech_synthesizer.speak_text_async(response_text).get()
# # Check result
# if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
#     print("Speech synthesized for text [{}]".format(text))
# elif result.reason == speechsdk.ResultReason.Canceled:
#     cancellation_details = result.cancellation_details
#     print("Speech synthesis canceled: {}".format(cancellation_details.reason))
#     if cancellation_details.reason == speechsdk.CancellationReason.Error:
#         print("Error details: {}".format(cancellation_details.error_details))


