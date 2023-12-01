# Food orders – speech to text to JSON – get a food order, use Azure AI Speech Cognitive service to transform 
# speech to text and then using Azure OpenAI GPT3.5 to extract a JSON object that represents the order and can be used to search in a Database
# In this case we are extracting restaurant order entities.
# try saying:  I would like to order a large burger which French fries and a large coke.

import azure.cognitiveservices.speech as speechsdk
from dotenv import load_dotenv
import pandas as pd
import utils
import openai
import utils

def voice_to_text():
    speech_config = speechsdk.SpeechConfig(subscription=utils.SPEECH_KEY, region=utils.SPEECH_REGION)
    speech_config.speech_recognition_language="en-US"

    audio_config = speechsdk.audio.AudioConfig(use_default_microphone=True)
    speech_recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config, audio_config=audio_config)

    print("Can I have your order please?")
    speech_recognition_result = speech_recognizer.recognize_once_async().get()

    if speech_recognition_result.reason == speechsdk.ResultReason.RecognizedSpeech:
        print("Text: {}".format(speech_recognition_result.text))
    return speech_recognition_result.text


def call_openai(text):
    template_prefix = """
    <|im_start|>system
    You are an assistant designed to extract entities from a food order transcript. Users will paste in a string of text and you will respond with entities you\'ve extracted from the text as a JSON object. Here\'s an example of your output format:
    [  {    main: {      type: \"vegan burger\",      size: \"large\",      cooking_degree: \"medium\",      toppings: [        {          type: \"lettuce\",          quantity: 1,          size: \"small\"        },        {          type: \"tomato\",          quantity: 1,          size: \"\"        },        {          type: \"onion\",          quantity: 2,          size: \"\"        }      ]    },drinks: {  type: \"pepsi cola\",  size: \"large\",  additionals: [{  type: \"ice\",  quantity: 1,  size: \"small\"},{  type: \"lemon\",  quantity: 1,  size: \"\"}  ]}  }]

    """
    template_sufix = "<|im_end|>\n<|im_start|>assistant"

    prompt = template_prefix + text + template_sufix
    response = openai.Completion.create(
        engine=utils.OPENAI_DEPLOYMENT_NAME,
        prompt=prompt,
        temperature=0,
        max_tokens=4096,
        top_p=0.95,
        frequency_penalty=0,
        presence_penalty=0,
        stop=["<|im_end|>"])
    response = response['choices'][0]['text']
    response = utils.remove_chars("\n", response)
    response=utils.start_after_string("Answer:", response)
    response=utils.remove_tail_tags("<|im_end|>", response)
    return response

if __name__ == "__main__":
    text = voice_to_text()
    response = call_openai(text)
    print("Order: \n")
    utils.pretty_print_json_string(response)

