# Imports the Google Cloud Translation library
import six
from google.cloud import translate_v2 as translate
import io
from tqdm import tqdm

from google.cloud.speech_v2 import SpeechClient
from google.cloud.speech_v2.types import cloud_speech

import os
import openai
openai.organization = "org-JaGJq15JH1OngBfov966SFaN"
openai.api_key = os.getenv("OPENAI_API_KEY")

PROJECT_ID = "my-project-98575-371210"


def translate_text(text):
    """Translates text into the target language.

    Target must be an ISO 639-1 language code.
    See https://g.co/cloud/translate/v2/translate-reference#supported_languages
    """


    translate_client = translate.Client()

    if isinstance(text, six.binary_type):
        text = text.decode("utf-8")

    # Text can also be a sequence of strings, in which case this method
    # will return a sequence of results for each text.
    result = translate_client.translate(text, target_language="en-US")

    print(u"Text: {}".format(result["input"]))
    print(u"Translation: {}".format(result["translatedText"]))
    print(u"Detected source language: {}".format(result["detectedSourceLanguage"]))
    return result["translatedText"]




def transcribe_streaming_v2(project_id, recognizer_name, audio_file):
    # print("transcribe_streaming_v2", project_id, recognizer_name, audio_file)
    # return
    # Instantiates a client
    client = SpeechClient()

    request = cloud_speech.CreateRecognizerRequest(
        parent=f"projects/{project_id}/locations/global",
        recognizer_id="recognizer-4",
        recognizer=cloud_speech.Recognizer(
            language_codes=["ru-RU"], model="latest_long"
        ),
    )

    # Creates a Recognizer
    # operation = client.create_recognizer(request=request)
    # recognizer = operation.result()

    # Reads a file as bytes
    with io.open(audio_file, "rb") as f:
        content = f.read()

    # In practice, stream should be a generator yielding chunks of audio data
    chunk_length = 15360
    stream = [
        content[start * chunk_length : (start+1) * chunk_length]
        for start in range(0, len(content) // chunk_length)
    ]
    audio_requests = (
        cloud_speech.StreamingRecognizeRequest(audio=audio) for audio in stream
    )

    recognition_config = cloud_speech.RecognitionConfig(auto_decoding_config={})
    streaming_config = cloud_speech.StreamingRecognitionConfig(
        config=recognition_config
    )
    config_request = cloud_speech.StreamingRecognizeRequest(
        recognizer=recognizer_name, streaming_config=streaming_config
    )

    def requests(config, audio):
        yield config
        for message in tqdm(audio):
            yield message

    # Transcribes the audio into text
    responses_iterator = client.streaming_recognize(
        requests=requests(config_request, audio_requests)
    )
    responses = []
    for response in responses_iterator:
        responses.append(response)
        for result in response.results:
            print("Transcript: {}".format(result.alternatives[0].transcript))

    return responses


def save_transcript(trans_resp, path):
    # print("save_transcript", trans_resp, path)
    # return
    transcripts = []
    for resp in trans_resp:
        for result in resp.results:
            alternative = result.alternatives[0]
            transcripts.append(f"{alternative.transcript}/{alternative.words[0].start_offset.seconds}/{alternative.words[0].end_offset.seconds}")

    with open(path, 'w') as f:
        f.write("\n".join(transcripts))

def parse_transcript(speakers_dict):
    # print("speakers_dict", speakers_dict)
    # return
    all_trans = []
    for dictor_name in speakers_dict:
        transcript_path = speakers_dict[dictor_name]
        with open(transcript_path) as f:
            transcript = f.read()
            transcript = transcript.split("\n")
        for item in transcript:
            line, start,end = item.split("/")
            start, end = float(start), float(end)
            all_trans.append((dictor_name, line, start,end))
    
    all_trans.sort(key=lambda x: x[2])
    return all_trans

def merge_transcripts(all_trans):
    new_transcript = []
    last_person = None
    for item in all_trans:
        person, transcript, start, end = item

        if person == last_person:
            new_transcript.append(transcript)
        else:
            new_transcript.append(f"\n{person}: {transcript}")
        last_person = person
    new_transcript = "".join(new_transcript)
    return new_transcript

def save_txt(translated_txt, path):
    with open(path, 'w') as f:
        f.write(translated_txt)

def get_summary(text):
    if len(text.split()) > 2500:
        raise Exception("Too long text")

    prompt = f"Here is a transcript of the meeting, make a summary: {text}"
    resp = openai.Completion.create(engine="text-davinci-003", prompt=prompt, max_tokens=500, temperature=0.9, top_p=1, frequency_penalty=0.0, presence_penalty=0.6)
    return resp.choices[0].text

def get_arrangements(text):
    if len(text.split()) > 2500:
        raise Exception("Too long text")

    prompt = f"Here is a transcript of the meeting, write about setted arrangements: {text}"
    resp = openai.Completion.create(engine="text-davinci-003", prompt=prompt, max_tokens=500, temperature=0.9, top_p=1, frequency_penalty=0.0, presence_penalty=0.6)
    return resp.choices[0].text

def get_insights(text):
    if len(text.split()) > 2500:
        raise Exception("Too long text")

    prompt = f"Here is a transcript of the meeting, provide meeting insights: {text}"
    resp = openai.Completion.create(engine="text-davinci-003", prompt=prompt, max_tokens=500, temperature=0.9, top_p=1, frequency_penalty=0.0, presence_penalty=0.6)
    return resp.choices[0].text

def get_risks(text):
    if len(text.split()) > 2500:
        raise Exception("Too long text")

    prompt = f"Here is a transcript of the meeting, tell about potential risks: {text}"
    resp = openai.Completion.create(engine="text-davinci-003", prompt=prompt, max_tokens=500, temperature=0.9, top_p=1, frequency_penalty=0.0, presence_penalty=0.6)
    return resp.choices[0].text


# if __name__ == "__main__":

#     # trans_resp = transcribe_streaming_v2(PROJECT_ID, 'projects/472813974978/locations/global/recognizers/recognizer-4', "stas.wav")
#     # save_transcript(trans_resp=trans_resp, path="stas_transcript.txt")

#     #trans_resp = transcribe_streaming_v2(PROJECT_ID, 'projects/472813974978/locations/global/recognizers/recognizer-4', "artem.wav")
#     #save_transcript(trans_resp=trans_resp, path="artem_transcript.txt")

#     # all_trans = parse_transcript()
#     # new_transcript = merge_transcripts(all_trans)
#     # translated_txt = translate_text("%".join(new_transcript.split("\n")))
#     # save_translated_txt(translated_txt, "translated_transcript.txt")

#     with open("translated_transcript.txt") as f:
#         translated_txt = f.read()
    
#     translated_txt = translated_txt.replace("&#39;", "")

#     print(f"Transcript have: {len(translated_txt.split())} words")

#     translated_txt = "\n".join(translated_txt.split("%"))

#     ok_words = translated_txt.split()[-2450:]
#     ok_words_len = sum([len(word) for word in ok_words]) + len(ok_words)

#     cutted_translated_txt = translated_txt[:ok_words_len]


    
#     # summary = get_summary(cutted_translated_txt)
#     # save_translated_txt(summary, "summary.txt")
    
#     # arrangement = get_arrangements(cutted_translated_txt)
#     # save_translated_txt(arrangement, "arrangements.txt")
    
#     # insights = get_insights(cutted_translated_txt)
#     # save_translated_txt(insights, "insights.txt")

#     risks = get_risks(cutted_translated_txt)
#     save_translated_txt(risks, "risks.txt")
    
#     1/0