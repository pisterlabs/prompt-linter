""" This module contains functions for generating ideas for youtube videos. """
from langchain.chains import LLMChain
from langchain.chains import SequentialChain
from langchain.prompts import PromptTemplate
from utils.customllm import GPTv1


def video_ideation(abstract):
    """
    Create a video script, title and thumbnail idea based on a short description.

    Args:
        abstract (str): Short description of the video.

    Returns:
        script (str): Script of the video.
        title (str): Title of the video.
        thumbnail (str): Thumbnail idea for the video.
    """
    llm = GPTv1()
    title_text = """You are an expert on creating youtube titles that maximize impressions and click through rate of a video, 
    based on a video script. Your titles are controversial and eye popping and they are not longer than 80 characters. 
    Create 10 distinct youtube titles for this short description, delimited by triple quotes. '''{script}''' """

    script_text = """You are an expert on creating scripts for youtube videos based on a topic, 
    which you infer from the description. Your script maximizes the viewer retention of 
    said video and is tailored in such a way that a beginner in this topic problem understanding it.
    You also include ideas for stock footage to use inside the script, these ideas are delimited by 
    square brackets. Create a script draft for a video with the following description, delimited
    by triple quotes. 

    SHORT DESCRIPTION:
    '''{abstract}'''
    """
    thumbnail_text = """You are an expert on creating ideas for youtube thumbnails that maximize the
    impressions and click through rate of a video. Your ideas are minimalistic, yet they are still catchy.
    Give me an idea for a thumbnail based on the following video script and title, both delimited by triple quotes.

    VIDEO TITLE:
    '''{title}'''

    VIDEO SCRIPT
    '''{script}'''
    """
    title_template = PromptTemplate(
        input_variables=["abstract"], template=script_text
    )
    script_chain = LLMChain(
        llm=llm, prompt=title_template, output_key="script"
    )
    script_template = PromptTemplate(
        input_variables=["script"], template=title_text
    )
    title_chain = LLMChain(llm=llm, prompt=script_template, output_key="title")
    thumb_template = PromptTemplate(
        input_variables=["title", "script"], template=thumbnail_text
    )
    thumb_chain = LLMChain(
        llm=llm, prompt=thumb_template, output_key="thumbnail"
    )

    overall_chain = SequentialChain(
        chains=[script_chain, title_chain, thumb_chain],
        input_variables=["abstract"],
        output_variables=["script", "title", "thumbnail"],
        verbose=False,
    )

    return overall_chain(abstract)
