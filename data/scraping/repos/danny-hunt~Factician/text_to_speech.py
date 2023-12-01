import json
import re
from typing import Literal, Optional, TypedDict
from anthropic import Anthropic, HUMAN_PROMPT, AI_PROMPT
from google.cloud import speech
from utils.cleaned_police_dodge import full_police_transcript
from utils.cleaned_trump import trump_transcript
from utils.wwf_article import wwf_article_text as wwf

GS_LINK = "gs://anthropic-paxmamn/Trump.wav" # doubles.wav

anthropic = Anthropic()


class Person(TypedDict):
    name: str
    color: str
    kind: Literal["interviewer", "interviewee"]

TagInfo = {
    2: {"name": "Donald Trump", "color": "red", "kind": "interviewee"},
    1: {"name": "Leslie Stahl", "color": "blue", "kind": "interviewer"},
    # 3: {"name": "Jeff", "color": "green", "kind": "interviewer"},
}

class FullTranscriptionUnit(TypedDict):
    word: str
    speaker_tag: int
    start_time: float
    end_time: float

FullTranscription = list[FullTranscriptionUnit]

class TranscriptionUnit(TypedDict):
    phrase: str
    speaker_tag: int
    start_time: float
    end_time: float

Transcription = list[TranscriptionUnit]

class Interruption(TypedDict):
    interrupter_tag: int
    time: float

class AnalysisDict(TypedDict):
    time_end: float
    transcription: Transcription
    interruptions: int
    evasiveness: Optional[int]  # 0-100
    friendly: Optional[int]  # 0-100
    questions: list[str]

CompletedAnalysis = list[AnalysisDict]


def collapse_transcription(transcription: Transcription, current_time: float) -> Transcription:
    output: Transcription = []
    current_word = ""
    current_speaker = 0
    current_start = 0.0
    current_end = current_time
    for word in transcription:
        if word["end_time"] > current_time:
            break
        if word["speaker_tag"] == current_speaker:
            current_word += word["word"] + " "
            current_end = word["end_time"]
        else:
            if current_speaker != 0:
                output.append({"word": current_word, "speaker_tag": current_speaker, "start_time": current_start, "end_time": current_end})
                current_word = word["word"] + " "
            current_speaker = word["speaker_tag"]
            current_start = word["start_time"]
            current_end = word["end_time"]
    output.append({"word": current_word, "speaker_tag": current_speaker, "start_time": current_start, "end_time": current_end})
    return output


def run_quickstart() -> Transcription:
    # Instantiates a client
    client = speech.SpeechClient()

    # The name of the audio file to transcribe
    gcs_uri = GS_LINK # "gs://cloud-samples-data/speech/brooklyn_bridge.raw"

    audio = speech.RecognitionAudio(uri=gcs_uri)

    diarization_config = speech.SpeakerDiarizationConfig(
        enable_speaker_diarization=True,
        min_speaker_count=2,
        max_speaker_count=3,
    )

    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        # sample_rate_hertz=16000,
        language_code="en-US",
        diarization_config=diarization_config,
    )

    # Detects speech in the audio file
    print("Waiting for operation to complete...")
    response = client.recognize(config=config, audio=audio)

    print(response)

    result = response.results[-1]

    words_info = result.alternatives[0].words

    output: Transcription = []

    # Printing out the output:
    for word_info in words_info:
        output.append({"word": word_info.word, "speaker_tag": word_info.speaker_tag, "start_time": word_info.start_time.total_seconds(), "end_time": word_info.end_time.total_seconds()})
        print(f"word: '{word_info.word}', speaker_tag: {word_info.speaker_tag}")
    return output

def transcript_to_prompt(transcript: Transcription) -> str:
    output = ""
    for phrase in transcript:
        # print(phrase)
        output += f"<{TagInfo[phrase['speaker_tag']]['name'] + ' (politician)' if TagInfo[phrase['speaker_tag']]['kind'] == 'interviewee' else '(interviewer)'}>\n"
        output += phrase["word"] + "\n"
        output += f"</{TagInfo[phrase['speaker_tag']]['name']}>\n"
    return output

def timed_transcript_to_prompt(transcript: Transcription) -> str:
    output = ""
    for phrase in transcript:
        # print(phrase)
        output += f"<{TagInfo[phrase['speaker_tag']]['name'] + ' (politician)' if TagInfo[phrase['speaker_tag']]['kind'] == 'interviewee' else '(interviewer)'}>\n"
        output += phrase["word"] +  "(" + str(phrase["end_time"] - phrase["start_time"]) + "s)" + "\n"
        output += f"</{TagInfo[phrase['speaker_tag']]['name']}>\n"
    return output


def detect_interviewee_attitude(transcript: Transcription, attribute: Literal['evasive', 'friendly'], contra_attribute: Optional[Literal['angry']]) -> Optional[int]:
    prompt = f"""I'm going to give you a transcript from a political interview and then I'm going to ask you some questions about it.
    <transcript>
    {transcript_to_prompt(transcript)}</transcript>
    Here is the first question: on a scale of 0-100 how {attribute} is the politician being interviewed? A 0 means {f'more like {contra_attribute}' if contra_attribute else f'not at all {attribute}'} and a 100 means extremely {attribute}.
    Return nothing but the numerical answer. Use JSON format with the key as "rating". If there is not enough information to answer then you may say so instead.

    Assistant:
    """
    anthropic = Anthropic()
    completion = anthropic.completions.create(
        model="claude-2",
        max_tokens_to_sample=300,
        prompt=f"{HUMAN_PROMPT} {prompt}",
    )
    print(completion.completion)
    match = re.search(r'[0-9]+', completion.completion)
    """
    Use JSON format with the keys as "rating"
    """
    if match:
        value = int(match.group())
        return value if value != 0 and value != 100 else None
    else:
        return None
    

def grammify_question(question: str) -> str:
    return question.capitalize() + ("?" if question[-1] != "?" else "")
    
def extract_questions(transcript: Transcription) -> [list[str], list[str]]:
    prompt = f"""I'm going to give you a transcript from a political interview and then I'm going to ask you some questions about it.
    <transcript>
    {transcript_to_prompt(transcript)}</transcript>
    What questions have been asked to the politican by the interviewer? Of those questions which have been answered?
    Use JSON format with the keys as "answered_questions" and "unanswered_questions". Their values are both lists of strings. Questions do not appear in both. 

    Assistant:
    """
    anthropic = Anthropic()
    completion = anthropic.completions.create(
        model="claude-2",
        max_tokens_to_sample=300,
        prompt=f"{HUMAN_PROMPT} {prompt}",
    )
    print(completion.completion)
    try:
        result_dict = json.loads(completion.completion)
    except Exception as e:
        print(e)
        return [[], []]
    return [
        list(map(grammify_question, result_dict["answered_questions"])) if "answered_questions" in result_dict and isinstance(result_dict["answered_questions"], list) else [],
        list(map(grammify_question, result_dict["unanswered_questions"])) if "unanswered_questions" in result_dict and isinstance(result_dict["unanswered_questions"], list) else []
    ]


def interruptions(transcript: Transcription) -> list[float]:
    prompt = f"""I'm going to give you a transcript from a political interview and then I'm going to ask you some questions about it.
    <transcript>
    {timed_transcript_to_prompt(transcript)}</transcript>
    How many times does the politician interrupt the interviewer? 
    Use JSON format with the key as "interruption_count". The value is an integer.

    Assistant:
    """
    anthropic = Anthropic()
    completion = anthropic.completions.create(
        model="claude-2",
        max_tokens_to_sample=300,
        prompt=f"{HUMAN_PROMPT} {prompt}",
    )
    print(completion.completion)
    match = re.search(r'[0-9]+', completion.completion)
    """
    Use JSON format with the keys as "rating"
    """
    if match:
        return int(match.group())
    else:
        return None
    

def detect_lies(transcript: Transcription) -> list[str]:
    # https://www.worldwildlife.org/pages/why-are-glaciers-and-sea-ice-melting
    article_text = wwf

    prompt = f"""I'm going to give you a truthful article about climate change and transcript from a political interview and then I'm going to ask you some questions about it.
    <article>
    {article_text}
    </article>
    <transcript>
    {timed_transcript_to_prompt(transcript)}</transcript>
    Does the politician constradict the article? If so, how?
    Use JSON format with the key as "explained_contradictions". The value is a list of strings

    Assistant:
    """
    anthropic = Anthropic()
    completion = anthropic.completions.create(
        model="claude-2",
        max_tokens_to_sample=1000,
        prompt=f"{HUMAN_PROMPT} {prompt}",
    )
    print(completion.completion)
    try:
        result_dict = json.loads(completion.completion)
    except Exception as e:
        print(e)
        return []
    return list(map(grammify_question, result_dict["explained_contradictions"])) if "explained_contradictions" in result_dict and isinstance(result_dict["explained_contradictions"], list) else [],


if __name__ == "__main__":
    # full_transcript = run_quickstart()
    # with open("trump_transcript.txt", "w") as f:
    #     f.write(json.dumps(full_transcript, indent=4))
    # print("transcribed")

    # full_transcript = full_police_transcript
    full_transcript = trump_transcript

    analysis: CompletedAnalysis = [
        {
            "time_end": 0,
            "transcription": [],
            "evasiveness": None,
            "friendly": None,
            "interruptions": 0,
            "answered_questions": [],
            "unanswered_questions": [],
            "contradictions": [],
        }
    ]

    for t in range(3, 63, 3):
        temp_transcription = collapse_transcription(full_transcript, t)
        evasiveness = detect_interviewee_attitude(temp_transcription, "evasive", None)
        friendly = detect_interviewee_attitude(temp_transcription, "plain-spoken", None)
        answered_questions, unanswered_questions = extract_questions(temp_transcription)
        interruptions_count = interruptions(temp_transcription)
        contradictions = detect_lies(temp_transcription)
        print(f"{evasiveness=}")
        print(f"{friendly=}")
        analysis.append({
            "time_end": t,
            "transcription": temp_transcription,
            "evasiveness": evasiveness,
            "friendly": friendly, # This is now plain-spokenness
            "interruptions": interruptions_count,
            "answered_questions": answered_questions,
            "unanswered_questions": unanswered_questions,
            "contradictions": contradictions,
        })



    # temp_transcription = collapse_transcription(full_transcript, 60)

    # evasiveness = detect_interviewee_attitude(temp_transcription, "evasive")
    # print(f"{evasiveness=}")

    with open("full_transcript.txt", "w") as f:
        f.write(json.dumps(analysis, indent=4))