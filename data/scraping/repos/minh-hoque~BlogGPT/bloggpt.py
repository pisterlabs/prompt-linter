# Purpose: Run the bloggpt agent

import logging
import os
import sys
import traceback
from pprint import pprint
from typing import List, Optional

import streamlit as st
from ansi2html import Ansi2HTMLConverter
from dotenv import load_dotenv
from langchain.agents import AgentType, Tool, initialize_agent
from langchain.chat_models import ChatOpenAI

from prompts.prompts import (
    BLOG_SECTION_AGENT_SYSTEM_PROMPT,
    OUTLINE_PROMPT,
    REWRITE_PROMPT,
    TOPIC_PROMPT,
)
from utils.logging_utils import StreamlitPrint
from utils.main_utils import (
    bprint,
    combine_drafts,
    generate_final_blog,
    gprint,
    rprint,
    split_outline_prompt,
)
from utils.web_utils import search_and_summarize_web_url

sys.stdout = StreamlitPrint()

# Load environment variables from .env file
load_dotenv(".")

# Use the environment variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

tools = [search_and_summarize_web_url]

chat_model = ChatOpenAI(
    model_name="gpt-4-0613",
    temperature=0,
    openai_api_key=OPENAI_API_KEY,
)

# https://github.com/hwchase17/langchain/issues/6025
blog_agent = initialize_agent(
    tools,
    chat_model,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    max_iterations=4,
)


def get_topic_context(topic: str) -> Optional[str]:
    """
    Get the search context for a given topic from the web.

    Args:
        topic: A string representing the topic.

    Returns:
        Relevant information retrieved from the web about the topic.
        If an error occurs during the search, returns None.
    """
    with st.expander(f"Search Results For: {topic}"):
        try:
            context = search_and_summarize_web_url(topic)
            st.write(context)
        except Exception as e:
            st.error(f"Error occurred while searching for {topic}: {e}")
            st.error(traceback.format_exc())
            raise e
    st.divider()
    return context


def generate_blog_section(
    header: str, blog_section: str, context: str
) -> Optional[str]:
    """
    Generate a blog section for a given header, blog_section and context.

    Args:
        header: A string representing the header.
        blog_section: A string representing the blog section.
        context: A string representing the context.

    Returns:
        A string representing the generated blog section.
        If an error occurs during the generation, returns None.
    """
    try:
        with st.expander(f"Generating Blog Section: {header}"):
            GENERATE_BLOG_SECTION_PROMPT = BLOG_SECTION_AGENT_SYSTEM_PROMPT.format(
                TOPIC_PROMPT=TOPIC_PROMPT,
                BLOG_SECTION_OUTLINE_PROMPT=blog_section,
                CONTEXT=context,
            )
            logging.debug(GENERATE_BLOG_SECTION_PROMPT)
            rprint(f"Generating Blog Section: {header}")
            generated_blog = blog_agent.run(GENERATE_BLOG_SECTION_PROMPT)
    except Exception as e:
        st.error(f"Error occurred while generating blog section {header}: {e}")
        st.error(traceback.format_exc())
        generated_blog = None
    return generated_blog


def save_blog_section(generated_blog: str, header: str, num_generated: int) -> None:
    """
    Save a generated blog section.

    Args:
        generated_blog: A string representing the generated blog.
        header: A string representing the header.
        num_generated: An integer representing the number of generated sections.

    Returns:
        None. If an error occurs during the saving process, logs the error.
    """
    try:
        with st.expander(f"Blog Section: {header}"):
            st.write(generated_blog)
        with open(f"outputs/draft_{num_generated}.md", "w") as f:
            f.write(generated_blog)
    except Exception as e:
        st.error(f"Error occurred while saving blog section {header}: {e}")
        st.error(traceback.format_exc())
    st.divider()


def combine_and_finalize_draft(generated_blogs: List[str]) -> None:
    """
    Combine and finalize the draft.

    Args:
        headers: A list of strings representing the headers.
        blog_sections: A list of strings representing the blog sections.

    Returns:
        None. If an error occurs during the process, logs the error.
    """
    try:
        with st.expander("Blog Draft"):
            rprint("Combining the generated blog sections into one")
            entire_draft = "\n\n".join(generated_blogs)
            st.write(entire_draft)

        # entire_draft = combine_drafts("outputs/")
        with st.expander("Final Blog"):
            rprint("Generating Final Blog")
            final_blog = generate_final_blog(entire_draft, TOPIC_PROMPT, OPENAI_API_KEY)
            gprint("Done!")
            st.write(final_blog)
    except Exception as e:
        logging.error(f"Error occurred while finalizing blog: {e}")
        st.error(traceback.format_exc())


@st.cache_data()
def run_bloggpt(topic_str: str, blog_outline: str) -> None:
    """
    Run bloggpt to generate a blog from provided topic and blog outline.
        topic_str: A string representing the topic.
        blog_outline: A string representing the blog outline.

    Returns:
        None.
    """
    topic = topic_str.split(":")[1].strip()
    context = get_topic_context(topic)
    headers, blog_sections = split_outline_prompt(blog_outline)
    generated_blogs = []
    for num_generated, (header, blog_section) in enumerate(zip(headers, blog_sections)):
        generated_blog = generate_blog_section(header, blog_section, context)
        if generated_blog is not None:
            generated_blogs.append(generated_blog)
            with st.expander(f"Blog Section: {header}"):
                st.write(generated_blog)
            # save_blog_section(generated_blog, header, num_generated)
    combine_and_finalize_draft(generated_blogs)


if __name__ == "__main__":
    run_bloggpt(TOPIC_PROMPT, OUTLINE_PROMPT)
