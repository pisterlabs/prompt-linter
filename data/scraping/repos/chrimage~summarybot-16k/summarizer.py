import os
from transcriber import transcribe_video
from utils import get_video_title
from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    AIMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.schema import AIMessage, HumanMessage, SystemMessage
from dotenv import load_dotenv
load_dotenv()

def get_summary_filename_and_path(video_title):
    """Get the summary filename and path."""
    summary_filename = video_title + "-summary.md"
    summary_path = os.path.join("summaries", summary_filename)
    return summary_filename, summary_path

def summarize_video(video_url: str):
    video_title = get_video_title(video_url)
    transcript_path = transcribe_video(video_url, video_title)
    with open(transcript_path, "r") as f:
        transcript = f.read()
    chat = ChatOpenAI(temperature=1.0,model="gpt-3.5-turbo-16k",openai_api_key=os.getenv("OPENAI_API_KEY"))
    system_message_file = "system_message.txt"
    if os.path.exists(system_message_file):
        with open(system_message_file, "r") as f:
            system_message_content = f.read()
    else:
        system_message_content = "You are a YouTube video summarizer. You will be provided with a transcript of the video. Please reply with a detailed summary of the video."
    messages = [
        SystemMessage(content=system_message_content),
        HumanMessage(content=f"Title: {video_title}\n\n{transcript}")
    ]
    response = chat(messages)
    summary = response.content
    from utils import ensure_directory_exists

    # Create summaries folder if it doesn't exist
    summaries_folder = "summaries"
    ensure_directory_exists(summaries_folder)
    # Save summary to file. We replace the transcript file's extension with .txt
    _, summary_path = get_summary_filename_and_path(video_title)
    with open(summary_path, "w") as f:
        f.write(summary)
    print("Summary completed.")
    return summary_path
