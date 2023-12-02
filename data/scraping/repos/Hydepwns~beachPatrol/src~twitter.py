from langchain.prompts import PromptTemplate
from twspace_dl.api import API
from twspace_dl.cookies import load_cookies
from twspace_dl.twspace import Twspace
from twspace_dl.twspace_dl import TwspaceDL
import json
import time
import random
import os
import subprocess
import openai
import threading
from langchain.schema import Document
from langchain.chat_models import ChatOpenAI
from langchain.chains.summarize import load_summarize_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv

load_dotenv()


def download_twitter_space_direct(space_url, cookie_file, output_format="/tmp/spaces/%(creator_id)s-%(id)s"):
    API.init_apis(load_cookies(cookie_file))
    twspace = Twspace.from_space_url(space_url)

    # Initialize TwspaceDL with a specific output format
    twspace_dl = TwspaceDL(twspace, output_format)

    # This will now use the custom output format you provided
    file_save_path = twspace_dl.filename + ".m4a"

    if (os.path.exists("/tmp/spaces/") == False):
        os.makedirs("/tmp/spaces/")

    try:
        twspace_dl.download()
        twspace_dl.embed_cover()
    except KeyboardInterrupt:
        print("Download Interrupted by user")
    finally:
        twspace_dl.cleanup()

    if os.path.exists(file_save_path):
        return file_save_path
    else:
        return None


def get_twitter_space_if_live(user_url, cookies_path) -> Twspace | None:
    API.init_apis(load_cookies(cookies_path))
    try:
        twspace = Twspace.from_user_avatar(user_url)
    except Exception as e:
        print(e)
        return None

    return twspace


def chunk_file_if_needed(file_path, max_size_mb=10):
    file_size_mb = os.path.getsize(file_path) / (1024 * 1024)

    if file_size_mb <= max_size_mb:
        # Return the original path in a list
        return [file_path]
    else:
        # Calculate segment time based on estimated file size
        duration = get_audio_duration(file_path)
        estimated_segment_time = int(duration * (max_size_mb / file_size_mb))

        # Create directory to store segments
        base_name = os.path.basename(file_path).rsplit('.', 1)[0]
        segments_dir = f"/tmp/spaces/{base_name}/segments"
        os.makedirs(segments_dir, exist_ok=True)

        # Use FFmpeg to split file into chunks. Note the usage of '-acodec mp3' to explicitly set the audio codec.
        cmd = (
            f"ffmpeg -i {file_path} -f segment -segment_time {estimated_segment_time} "
            f"-acodec libmp3lame -b:a 192k {segments_dir}/segment%09d.mp3"
        )

        os.system(cmd)

        # Return the list of chunk file paths
        return [os.path.join(segments_dir, f) for f in os.listdir(segments_dir) if f.startswith("segment")]


def get_audio_duration(file_path):
    # Using ffprobe to get the duration of the audio
    cmd = f"ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 {file_path}"
    result = subprocess.check_output(cmd, shell=True)
    return float(result)


def transcribe_segments(segments, prompt):
    """
    Transcribe the audio segments using OpenAI's Whisper API.

    :param segments: List of paths to audio segment files.
    :param prompt: A prompt to provide context for the transcription.
    :return: A dictionary with segment paths as keys and their transcriptions as values.
    """

    openai.api_key = os.getenv("OPENAI_API_KEY")
    transcript = ""
    for segment in segments:
        with open(segment, "rb") as audio_file:
            res = openai.Audio.transcribe("whisper-1", audio_file)
            transcript += str(res['text'])

    return transcript


summary_template = """
    You are an analytics professional at Lido Finance, a Liquid Staking protocol for Ethereum. You are given a transcript of Twitter Spaces in the crypto/web3 space that may or may not be related to Lido.
    Given the transcript, you are writing structured notes in markdown format. Think of your notes as key takeaways, TLDRs, and executive summaries.

    Your notes should be concise, detailed, and structured by topics. You know what information is especially important, and what is less important.

    Here is the transcript:
    {text}
    
    YOUR NOTES:"""

refine_summary_template = """
    You are an analytics professional at Lido Finance, a Liquid Staking protocol for Ethereum. You are given a transcript of Twitter Spaces in the crypto/web3 space that may or may not be related to Lido.
    Given the transcript, you are refining structured notes in markdown format. Think of your notes as key takeaways, TLDRs, and executive summaries.

    Here is the existing note:
    {existing_answer}
    
    We have the opportunity to refine the existing note (only if needed) with some more context below:
    -----
    {text}
    -----
    
    Given the new context, refine the original note to make it more complete.
    If the context isn't useful, return the original summary.

    Your notes should be concise, detailed, and structured by topics. You know what information is especially important, and what is less important.

    Use markdown formatting to its fullest to produce visually appealing, structured notes.
    """

summary_prompt = PromptTemplate.from_template(summary_template)
refine_summary_prompt = PromptTemplate.from_template(refine_summary_template)


def summarize_transcript(transcript):
    """
    Summarizes a transcript.

    This function splits the transcript into chunks, creates a `ChatOpenAI` instance, loads a summarization chain, and applies the chain to the chunks.
    If an error occurs, it prints the error message and returns a string indicating an error.

    Args:
        transcript (str): The transcript to summarize.

    Returns:
        str: The summarized transcript or an error message.

    """
    try:
        doc = Document(page_content=transcript)

        # Summarize the document, splitting it into chunks of 40,000 characters with 500 characters of overlap.
        # model_name can be any model from https://huggingface.co/models?filter=summarization, chatgpt4 would be a good choice
        # temperature is a parameter for the model that controls how "creative" the model is. 0 is the most conservative, 1 is the most creative.
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=40000, chunk_overlap=500, length_function=len, is_separator_regex=False)
        docs = text_splitter.split_documents([doc])
        llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo-16k")
        chain = load_summarize_chain(llm, chain_type="refine",
                                     question_prompt=summary_prompt,
                                     refine_prompt=refine_summary_prompt,
                                     return_intermediate_steps=True,
                                     input_key="input_documents",
                                     output_key="output_text")

        result = chain({"input_documents": docs}, return_only_outputs=True)

        return result["output_text"]

    except Exception as e:
        print(f"Error summarizing transcript: {e}")
        return "Error summarizing transcript"

# Define a template for generating an executive summary. Feel free to edit.
executive_template = """
    Given the summary of a Twitter Space:
    {text}

    Generate an extremely brief executive summary for Lido executives. It should be concise, focused, and only contain information relevant to Lido Finance.
    """

refine_executive_template = """
    You are an analytics professional at Lido Finance, a Liquid Staking protocol for Ethereum. You are given a summary of Twitter Spaces in the crypto/web3 space that may or may not be related to Lido.
    Given the summary, you are refining an executive summary in markdown format. Think of your notes as key takeaways, TLDRs, and executive summaries.

    Here is the existing executive summary:
    {existing_answer}

    We have the opportunity to refine the existing executive summary (only if needed) with some more context below:
    -----
    {text}
    -----

    Your updated executive summary:"""

executive_prompt = PromptTemplate.from_template(executive_template)
refine_executive_prompt = PromptTemplate.from_template(
    refine_executive_template)


def get_executive_summary(summary):
    try:
        doc = Document(page_content=summary)
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=40000, chunk_overlap=500, length_function=len, is_separator_regex=False)
        docs = text_splitter.split_documents([doc])
        llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo-16k")
        chain = load_summarize_chain(llm, chain_type="refine",
                                     question_prompt=executive_prompt,
                                     refine_prompt=refine_executive_prompt,
                                     return_intermediate_steps=True,
                                     input_key="input_documents",
                                     output_key="output_text")

        result = chain({"input_documents": docs}, return_only_outputs=True)
        return result["output_text"]
    except Exception as e:
        print(f"Error generating executive summary: {e}")
        return "Error generating executive summary"


def process_twitter_space(space_url, cookies_path):
    transcript_location = download_twitter_space_direct(
        space_url, cookies_path)
    chunks = chunk_file_if_needed(transcript_location)
    transcript = transcribe_segments(
        chunks, "Twitter Space about Crypto, Web3, Liquid Staking, and Lido Finance")
    summary = summarize_transcript(transcript)
    executive_summary = get_executive_summary(summary)

    return {'space_url': space_url, 'exec_sum': executive_summary, 'notes': summary}
