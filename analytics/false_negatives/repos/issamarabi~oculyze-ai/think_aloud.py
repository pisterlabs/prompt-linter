import os
import openai
from openai import OpenAI
import whisper
from typing import List, Dict
openai.api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI()


def transcribe_audio(audio_files: List[str]) -> List[str]:
    model = whisper.load_model("large")
    transcriptions = [model.transcribe(audio_file) for audio_file in audio_files]
    return transcriptions


# def think_aloud_minutes(transcription: str) -> dict:
#     usability_issues = issues_extraction(transcription)
#     # actionable_insights = action_extraction(usability_issues, transcription)
#     return {
#         'usability_issues': usability_issues,
#         # 'actionable_insights': actionable_insights,  
#     }


def issues_extraction(transcription: str, website_name: str, tasks: str) -> str:
    response = client.chat.completions.create(
        model="gpt-4-1106-preview",
        temperature=0,
        messages=[
            {
                "role": "system",
                "content": f"""
                You are an AI with expertise in HCI and UX research and design and identification of usabilitiy issues of websites.
                Your task is to identify the main usability issues of a website from the following 
                transcripts of think-aloud data from a usability test of the website {website_name}, where they were directed to complete the following
                tasks: {tasks}. Based on the transcript provided, please identify and list the usability issues that users experienced while navigating or 
                interacting with the website. Focus on points where users felt confused, frustrated, or encountered problems. After conducting the analysis,
                your goal is to provide a clear list of issues that can help designers understand areas that need improvement. For each issue, referece 
                precisely where in the transcription it was mentioned or inferred.
                """
            },
            {
                "role": "user",
                "content": transcription
            }
        ]
    )
    return response.choices[0].message.content


# def action_extraction(issues: str, transcription: str) -> str:
#     response = client.chat.completions.create(
#         model="gpt-4",
#         temperature=0,
#         messages=[
#             {
#                 "role": "system",
#                 "content": "As an AI with expertise in HCI, your task is to identify the main actionable insights to give designers of a website from the following transcription from a usability test of a website, as well as the previously extracted usability issues. Given the extracted usability issues and the transcription, please provide actionable insights or recommendations that the designers can implement to improve the website's usability. These should be practical steps or changes that can address the identified problems. Your goal is to help designers take effective actions to enhance the user experience."
#             },
#             {
#                 "role": "user",
#                 "content": f"""
#                 Issues extracted: {issues}
#                 transcription: {transcription}
#                 """
#             }
#         ]
#     )
#     return response.choices[0].message.content


def format_arr(arr: List[str], data: str) -> str:
    """
    Converts a single transcript into a readable string format for LLM processing.
    """
    str = ""
    for i, item in enumerate(arr):
        str += f"{data} {i+1}: {item}\n"
    return str

def summarize_insights(analyses: str, website: str, tasks: str) -> str:
    """
    transcripts: list of transcripts (combined output of multiple calls to think_aloud_minutes())
    """

    # Generate a summary based on all transcripts
    response = client.chat.completions.create(
        model="gpt-4-1106-preview",
        temperature=0,
        messages=[
            {
                "role": "system",
                "content": f"""
                You are a superintelligent AI with speialist expertise in HCI and UX research and design.
                You are given a collection of analyses of usability studies for the website {website}, where participants conducted some tasks: {tasks}.
                (one analysis per participant)
                Each analysis contains a list of usability issues identified from a think-aloud session.
                Your task is to summarize these analyses based on the most significant conclusions accross analyses.
                For each issue, reference precisely where in the think-aloud transcript it was mentioned or inferred.
                """
            },
            {
                "role": "user",
                "content": analyses
            },
        ],
    )

    return response.choices[0].message.content


def process_log(log, website, tasks):
    response = client.chat.completions.create(
        model="gpt-4-1106-preview",
        temperature=0,
        messages=[
            {
                "role": "system",
                "content": f"""
                You are an AI with expertise in HCI and UX research and design and identification of usabilitiy issues of websites.
                You will be given website interaction logs from a usability study on the website {website}, where participants were asked to complete the
                following tasks: {tasks}.
                Analyze the following website interaction logs to identify potential usability issues and user behavior patterns.
                Consider the user's navigation flow, interaction with various elements, points of hesitation or confusion, and any rapid or repetitive actions that might indicate frustration.

                            Based on these logs:
                            1. Identify patterns of user behavior.
                            2. Highlight deviations from typical behavior indicating usability issues.
                            3. Pinpoint areas of potential confusion or trouble spots.

                             For each issue, referece precisely what part of the logs points to it.
                            """
            },
            {
                "role": "user",
                "content": log
            },
        ],
    )

    return response.choices[0].message.content




# def abstract_summary_extraction(transcription):
#     response = client.chat.completions.create(
#         model="gpt-4",
#         temperature=0,
#         messages=[
#             {
#                 "role": "system",
#                 "content": "You are a highly skilled AI trained in language comprehension and summarization. I would like you to read the following transcription from a usability test of a website and summarize it into a concise abstract paragraph. Aim to retain the most important points, providing a coherent and readable summary that could help a person understand the main points of the discussion without needing to read the entire text. Please avoid unnecessary details or tangential points."
#             },
#             {
#                 "role": "user",
#                 "content": transcription
#             }
#         ]
#     )
#     return response.choices[0].message.content


# def key_points_extraction(transcription):
#     response = client.chat.completions.create(
#         model="gpt-4",
#         temperature=0,
#         messages=[
#             {
#                 "role": "system",
#                 "content": "You are a proficient AI with a specialty in distilling information into key points. Based on the following transcription from a usability test of a website, identify and list the main points that were discussed or brought up. These should be the most important ideas, findings, or topics that are crucial to the essence of the discussion. Your goal is to provide a list that someone could read to quickly understand what was talked about."
#             },
#             {
#                 "role": "user",
#                 "content": transcription
#             }
#         ]
#     )
#     return response.choices[0].message.content


# def sentiment_analysis(transcription):
#     response = client.chat.completions.create(
#         model="gpt-4",
#         temperature=0,
#         messages=[
#             {
#                 "role": "system",
#                 "content": "As an AI with expertise in language and emotion analysis, your task is to analyze the sentiment of the following transcription from a usability test of a website. Please consider the overall tone of the discussion, the emotion conveyed by the language used, and the context in which words and phrases are used. Indicate whether the sentiment is generally positive, negative, or neutral, and provide brief explanations for your analysis where possible."
#             },
#             {
#                 "role": "user",
#                 "content": transcription
#             }
#         ]
#     )
#     return response.choices[0].message.content