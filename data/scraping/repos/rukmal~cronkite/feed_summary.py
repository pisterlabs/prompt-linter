from langchain import OpenAI
from langchain.chains.summarize import load_summarize_chain
from langchain.docstore.document import Document
from langchain.prompts import PromptTemplate
from typing import Dict, List
import logging
import time


# Summarization parameters
OPENAI_MODEL_NAME: str = "text-davinci-003"
OPENAI_MODEL_TEMPERATURE: float = 0.0  # 0 is fully deterministic, 1 is most random
OPENAI_MODEL_MAX_TOKENS: int = 1200  # lanchain automatically sets to max for OPENAI_MODEL_NAME
INITIAL_PROMPT_TEMPLATE: str = (
    "You are a professional news reporter."
    "Use the following article summary to create an professional summarized news report for a business executive."
    "You must group relevant information together into a single bullet point."
    "Related topics must be grouped together under the a relevant title."
    "Make sure each title has at least 3 bullet points in each category."
    "The content must have the following form"
    """
    # <title>
    - <bullet point 1>
    - <bullet point 2>

    # <title>
    - <bullet point 1>
    - <bullet point 2>
    """
    "The summary of the article is: `{text}`"
    "AN EXECUTIVE SUMMARY OF THE NEWS, IN BULLET POINT LIST WITH ONLY COMPLETE BULLET POINTS:"
)
REFINE_PROMPT_TEMPLATE: str = (
    "You are a professional news reporter."
    "Your job is to produce a professional summarized news report for a business executive."
    "You must group relevant information together into a single bullet point."
    "We have provided an existing executive summary of related news up to a point:"
    "`{existing_answer}`"
    "The content must have the following form"
    """
    # <title>
    - <bullet point 1>
    - <bullet point 2>

    # <title>
    - <bullet point 1>
    - <bullet point 2>
    """
    "We have the opportunity to refine the concise summary (only if needed) with some more context from an additional article summary below."
    "It may or may not be relevant."
    "Ignore any articles about subscription offers, and not being able to access the website d ue to a paywall."
    "Article:"
    "`{text}`"
    "Given the new content, refine the existing executive summary."
    "If the new content isn't helpful, return the original executive summary."
    "A COMPLETE EXECUTIVE SUMMARY OF THE NEWS, IN BULLET POINT LIST WITH ONLY COMPLETE BULLET POINTS:"
)

# Summarization helpers
initial_prompt = PromptTemplate(template=INITIAL_PROMPT_TEMPLATE, input_variables=["text"])
refine_prompt = PromptTemplate(template=REFINE_PROMPT_TEMPLATE, input_variables=["existing_answer", "text"])


def executive_summary(articles_data: List[Dict[str, object]], openai_api_key: str) -> Dict[str, object]:
    tic = time.time()

    logging.debug("Summarizing feed of article summaries", {"n": len(articles_data)})

    # Building langchain Documents with each of the summaries
    summary_docs = [
        Document(page_content="Title: `" + i["title"] + "` Content: `" + i["summary"] + "`")
        for i in articles_data
        if i is not None
    ]

    # Initialize OpenAI LLM with langchain
    llm = OpenAI(
        model_name=OPENAI_MODEL_NAME,
        temperature=OPENAI_MODEL_TEMPERATURE,
        max_tokens=OPENAI_MODEL_MAX_TOKENS,
        openai_api_key=openai_api_key,
    )

    # Creating langchain summarize chain
    summarize_chain = load_summarize_chain(
        llm=llm,
        chain_type="refine",
        return_intermediate_steps=True,
        question_prompt=initial_prompt,
        refine_prompt=refine_prompt,
    )

    # Building news summary
    newsfeed_summary = summarize_chain(
        inputs={"input_documents": summary_docs},
        return_only_outputs=False,
    )

    logging.debug(
        "Successfully created executive summary",
        {
            "n": len(articles_data),
            "time": time.time() - tic,
        },
    )

    return newsfeed_summary


def executive_summary_map_reduce(articles_data: List[Dict[str, object]], openai_api_key: str) -> Dict[str, object]:
    map_prompt_template = """Summarize the following news article summary to a single bullet point. Make it shorthand, and as information dense and concise as possible. Include the most relevant and impactful information from the article. It be a short sentence.
    Article: `{text}`
    BULLET POINT SUMMARY:
    """

    reduce_prompt_template = """You are a professional news reporter. Your job is to create a bullet point executive summary of the news.
    You have the opportunity to refine the concise summary (only if needed) with some more context from an additional article summary below. It may or may not be relevant. Ignore any articles about subscription offers, and not being able to access the website due to a paywall.
    Make the titles are high-level as possible. Titles like "Economy", "International Relations", and "Technology" are good. Come up with more relevant high-level titles.
    The structure should be:
    ```
    # <title 1>
    - <bullet point 1>
    - <bullet point 2>

    # <title 2>
    - <bullet point 1>
    - <bullet point 2>
    ```
    Article Summary: {text}
    Given the new content, refine the existing executive summary. If the new content isn't helpful, return the original executive summary.
    BULLET POINT EXECUTIVE SUMMARY WIT HTITLES:"""

    tic = time.time()

    logging.debug("Summarizing feed of article summaries", {"n": len(articles_data)})

    # Building langchain Documents with each of the summaries
    summary_docs = [
        Document(page_content="Title: `" + i["title"] + "` Content: `" + i["summary"] + "`")
        for i in articles_data
        if i is not None
    ]

    # Initialize OpenAI LLM with langchain
    llm = OpenAI(
        model_name=OPENAI_MODEL_NAME,
        temperature=OPENAI_MODEL_TEMPERATURE,
        max_tokens=OPENAI_MODEL_MAX_TOKENS,
        openai_api_key=openai_api_key,
        batch_size=1,
    )

    # Creating langchain summarize chain
    summarize_chain = load_summarize_chain(
        llm=llm,
        chain_type="map_reduce",
        return_intermediate_steps=True,
        map_prompt=PromptTemplate(template=map_prompt_template, input_variables=["text"]),
        combine_prompt=PromptTemplate(template=reduce_prompt_template, input_variables=["text"]),
        verbose=True,
    )

    # Building news summary
    llm_result = summarize_chain(inputs={"input_documents": summary_docs}, return_only_outputs=False)

    return llm_result
