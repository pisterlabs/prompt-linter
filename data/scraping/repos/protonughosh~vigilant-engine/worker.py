import openai
import requests

openai.api_key = "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"


def speech_to_text(audio_binary):
    return None


def text_to_speech(text, voice=""):
    # Set up Watson Text to Speech HTTP Api url
    base_url = 'https://sn-watson-tts.labs.skills.network'
    api_url = base_url + '/text-to-speech/api/v1/synthesize?output=output_text.wav'
    # Adding voice parameter in api_url if the user has selected a preferred voice
    if voice != "" and voice != "default":
        api_url += "&voice=" + voice
    # Set the headers for our HTTP request
    headers = {
        'Accept': 'audio/wav',
        'Content-Type': 'application/json',
    }
    # Set the body of our HTTP request
    json_data = {
        'text': text,
    }
    [{
	"resource": "/c:/Users/proto/Downloads/chatapp-with-voice-and-openai/worker.py",
	"owner": "_generated_diagnostic_collection_name_#2",
	"severity": 8,
	"message": "\"{\" was not closed",
	"source": "Pylance",
	"startLineNumber": 24,
	"startColumn": 17,
	"endLineNumber": 24,
	"endColumn": 18
}]
    # Send a HTTP Post reqeust to Watson Text to Speech Service
    response = requests.post(api_url, headers=headers, json=json_data)
    print('text to speech response:', response)
    return response.content


def openai_process_message(user_message):
    # Set the prompt for OpenAI Api
    prompt = "\"Act like a personal assistant. You can respond to questions, translate sentences, summarize news, and give recommendations. " + user_message + "\""
    # Call the OpenAI Api to process our prompt
    openai_response = openai.Completion.create(model="text-davinci-003", prompt=prompt,max_tokens=4000)
    print("openai response:", openai_response)
    # Parse the response to get the response text for our prompt
    response_text = openai_response.choices[0].text
    return response_text