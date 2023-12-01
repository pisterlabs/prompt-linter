# -*- coding: utf-8 -*-
"""Services for the user app."""
import base64
import os
import re
from io import BytesIO

import openai
from google.cloud import texttospeech

openai.api_key = os.environ.get("OPENAI_API_KEY")


def read_file(file_path):
    """
    Reads the contents of a file and returns them as a string.

    Args:
        file_path (str): The path to the file to be read.

    Returns:
        str: The contents of the file.
    """
    with open(file_path, "r", encoding="utf-8") as file:
        return file.read()


def chat_gpt(question, answer):
    """
    Sends a message to OpenAI's GPT-3 model and returns the response.

    This function uses the OpenAI API to send a message
    and receive a response from the GPT-3 model.
    The message includes the system prompt, an explanation of the STAR method,
    and the `question` and `answer` parameters that are passed to the function.

    The purpose of this function is to provide constructive criticism
    to people who are answering interview questions.
    The response from the GPT-3 model will aim to be detailed and encouraging,
    and will describe how to use the STAR method every time it is referenced.

    Parameters:
    question (str): The interview question to be answered.
    answer (str): The answer to the interview question.

    Returns:
    str: The response from the GPT-3 model, which provides
    constructive criticism and feedback on the answer.

    Example:
    >>> chat_gpt("Tell me about a time when you achieved a goal that you initially thought was out of reach.", "I achieved a goal by working hard.") # noqa
    "Good effort, but try using the STAR method to structure your answer. Explain the situation, task, action, and result of the situation you're describing. # noqa
    This will make your answer more comprehensive and easier for the interviewer to understand."
    """

    system_message = """
    You are JudgeGPT. You provide constructive criticism to people who are answering interview questions. Be detailed and encouraging. Describe how to use the STAR method every time you reference it. # noqa
    """
    # from UMSI CDO resources https://docs.google.com/document/d/16HY__RHplZMBGhWmM6OavMvXXwiuNMxSPsfGCTIKd-o/edit
    # only UofM students can access this document
    intro_example = read_file("interview_simulator/user/prompts/intro_example.txt")
    # from https://www.themuse.com/advice/star-interview-method
    star_example = read_file("interview_simulator/user/prompts/star_example.txt")
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": intro_example},
            {
                "role": "user",
                "content": star_example,
            },
            {
                "role": "assistant",
                "content": question,
            },
            {"role": "user", "content": answer},
        ],
        max_tokens=1000,
        temperature=0.34,
    )
    return response["choices"][0]["message"]["content"].strip()


def gpt_questions(resume, job_description):
    """

    Uses OpenAI's GPT-3 model to generate interview questions for a job
    candidate based on their resume and a job listing.

    Args:
    - resume (str): A string containing the job candidate's resume.
    - job_description (str): A string containing the job listing for the position the candidate is applying for. # noqa

    Returns:
    - A dictionary containing the generated interview questions and their corresponding audio transcripts.
      The keys are:
        - 'intro': The introductory statement made by the hiring manager.
        - 'intro_audio': The audio transcript for the introductory statement.
        - 'question_1': The first interview question.
        - 'question_1_audio': The audio transcript for the first interview question.
        - 'question_2': The second interview question.
        - 'question_2_audio': The audio transcript for the second interview question.
        - 'question_3': The third interview question.
        - 'question_3_audio': The audio transcript for the third interview question.

    Example Usage:
    ```
    resume = "John Doe is a software engineer with 5 years of experience."
    job_description = "We are looking for a software engineer with experience in Python and Django."
    interview_questions = gpt_questions(resume, job_description)
    print(interview_questions)
    ```
    """
    system_message = """
    You are a Hiring Manager. You work at the company who listed the job description, and you are interviewing a candidate with the resume above. # noqa

    Briefly introduce yourself as Alex. Then generate 3 interview questions related to both the job description and the candidate's resume. # noqa

    Clearly delineate the introduction with # Introduction, and the questions with # Question X out of 3. Use the candidate’s name when appropriate. # noqa
    """
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": f"Resume:\n-----\n{resume}"},
            {
                "role": "user",
                "content": f"Job description for listing your company made:\n---\n{job_description}\n\n",
            },
        ],
        max_tokens=1000,
        temperature=0.34,
    )
    intro, question_1, question_2, question_3 = tuple(
        map(
            lambda x: x.replace("\n", "").strip(),
            filter(
                lambda x: len(x) > 0,
                re.split(
                    r"#\s.*\n", response["choices"][0]["message"]["content"].strip()
                ),
            ),
        )
    )
    intro += " To start us off, tell me a bit about yourself."
    intro_a = text_to_speech(intro)
    question_1_a = text_to_speech(question_1)
    question_2_a = text_to_speech(question_2)
    question_3_a = text_to_speech(question_3)
    return {
        "intro": intro,
        "intro_audio": intro_a,
        "question_1": question_1,
        "question_1_audio": question_1_a,
        "question_2": question_2,
        "question_2_audio": question_2_a,
        "question_3": question_3,
        "question_3_audio": question_3_a,
    }


def text_to_speech(text):
    """

    Uses Google Cloud Text-to-Speech to convert the input text into # noqa
    speech and returns the audio data in base64 encoded format.

    Args:
    - text (str): The text to convert into speech.

    Returns:
    - A base64 encoded string containing the audio data of the synthesized speech.

    Example Usage:
    ```
    speech = text_to_speech("Hello, world!")
    print(speech)
    ```
    """
    client = texttospeech.TextToSpeechClient()
    input_text = texttospeech.SynthesisInput(text=text)
    voice = texttospeech.VoiceSelectionParams(
        language_code="en-US", name="en-US-Neural2-J"
    )
    audio_config = texttospeech.AudioConfig(
        audio_encoding=texttospeech.AudioEncoding.MP3
    )

    response = client.synthesize_speech(
        input=input_text, voice=voice, audio_config=audio_config
    )
    return base64.b64encode(response.audio_content).decode("utf-8")


class NamedBytesIO(BytesIO):
    """An in-memory file-like object that has a name attribute."""

    def __init__(self, data, name):
        """Initialize the object with the given data and name."""
        super().__init__(data)
        self.name = name


def transcribe_audio_with_whisper(audio_data):
    """

    Uses the OpenAI Whisper ASR API to transcribe the # noqa
    input audio data and returns the transcription.

    Args:
    - audio_data (bytes): The audio data to transcribe, in bytes format.

    Returns:
    - A string containing the transcription of the input audio.

    Example Usage:
    ```
    with open("audio.webm", "rb") as f:
        audio_data = f.read()
    transcription = transcribe_audio_with_whisper(audio_data)
    print(transcription)
    ```
    """
    # Create an in-memory file object from the audio data
    audio_file = NamedBytesIO(audio_data, "audio.webm")
    # Send the audio file to the Whisper ASR API
    response = openai.Audio.transcribe("whisper-1", audio_file)
    # Extract the transcription from the response
    transcription = response["text"].strip()
    return transcription
