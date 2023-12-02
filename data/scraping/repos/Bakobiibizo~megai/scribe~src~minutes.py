import os
import datetime
import openai
from dotenv import load_dotenv
import loguru

logger = loguru.logger

load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")


def extract_info(transcription, audio_file_name):
    
    

def meeting_minutes(transcription, audio_file_name):
    """
    Generates a meeting minutes document based on a transcription.
    :param transcription: The text transcription of the meeting.
    :type transcription: str
    :return: A dictionary containing the meeting minutes information.
    :rtype: dict
    """
    roles_and_instructions = [
        (
            "abstract_summary_extraction\n",
            "You are a highly skilled AI trained in language comprehension and summarization. I would like you to "
            "read the following text and summarize it into a concise abstract paragraph. Aim to retain the most "
            "important points, providing a coherent and readable summary that could help a person understand the "
            "main points of the discussion without needing to read the entire text. Please avoid unnecessary details "
            "or tangential points.\n\n",
        ),
        (
            "key_points_extraction\n",
            "You are a proficient AI with a specialty in distilling information into key points. Based on the "
            "following text, identify and list the main points that were discussed or brought up. These should be "
            "the most important ideas, findings, or topics that are crucial to the essence of the discussion. Your "
            "goal is to provide a list that someone could read to quickly understand what was talked about.\n\n",
        ),
        (
            "action_item_extraction\n",
            "You are an AI expert in analyzing conversations and extracting action items. Please review the text "
            "and "
            "identify any tasks, assignments, or actions that were agreed upon or mentioned as needing to be done. "
            "These could be tasks assigned to specific individuals, or general actions that the group has decided to "
            "take. Please list these action items clearly and concisely.\n\n",
        ),
        (
            "sentiment_analysis\n",
            "As an AI with expertise in language and emotion analysis, your task is to analyze the sentiment of the "
            "following text. Please consider the overall tone of the discussion, the emotion conveyed by the "
            "language used, and the context in which words and phrases are used. Indicate whether the sentiment is "
            "generally positive, negative, or neutral, and provide brief explanations for your analysis where "
            "possible.\n\n",
        ),
    ]
    minutes = {
        "filename": f"{audio_file_name}\n",
        "datetime": f"{datetime.datetime.now().isoformat(timespec='seconds')}\n",
    }
    for role, instruction in roles_and_instructions:
        minutes[role] = extract_info(instruction, transcription)

    return minutes
