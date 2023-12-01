from langchain import OpenAI
from langchain.chains.summarize import load_summarize_chain
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
import logging
import tiktoken
import time



# Summarization parameters
OPENAI_MODEL_NAME: str = "text-curie-001"
OPENAI_TOKENIZER_NAME: str = "cl100k_base"
OPENAI_MODEL_TEMPERATURE: float = 0.0  # 0 is fully deterministic, 1 is most random
OPENAI_MODEL_MAX_TOKENS: int = 500  # langchain automatically sets to max for OPENAI_MODEL_NAME
CHUNK_TOKEN_SIZE: int = 800
CHUNK_TOKEN_OVERLAP: int = 100
INITIAL_PROMPT_TEMPLATE: str = (
    "Write an extremely concise bullet point summary of the following article, in bullet points."
    "Use as few words as possible."
    "Only include any information that relates to the title of the article."
    "Ignore any information that appear to be website artifacts."
    "The title of the article is `{title}`"
    "Article:"
    "`{text}`"
    "CONCISE ARTICLE:"
)
REFINE_PROMPT_TEMPLATE: str = (
    "Your job is to produce a very concise bullet point final summary."
    "Include the most important information from the article."
    "We have provided an existing concise summary up to a certain point:"
    "`{existing_answer}`"
    "We have the opportunity to refine the concise summary (only if needed) with some more context below."
    "It may or may not be relevant."
    "`{text}`"
    "Given the new context, refine the original summary in a concise manner."
    "If the context isn't helpful, return the original concise summary."
    "Concisely summarize just the main points of the article in bullet points, and don't include any of the context or other irrelevant information.\n"
    "CONCISE ARTICLE SUMMARY:"
)


# Summarization helpers
encoder = tiktoken.get_encoding(OPENAI_TOKENIZER_NAME)
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_TOKEN_SIZE, chunk_overlap=CHUNK_TOKEN_OVERLAP, length_function=lambda x: len(encoder.encode(x))
)
initial_prompt = PromptTemplate(template=INITIAL_PROMPT_TEMPLATE, input_variables=["text", "title"])
refine_prompt = PromptTemplate(template=REFINE_PROMPT_TEMPLATE, input_variables=["existing_answer", "text"])


def multiple_bullet_summary(article_data: dict, openai_api_key: str) -> dict:
    """Create a multi-bullet point summary of an article, using the refine method with the OPENAI_MODEL_NAME model.

    Arguments:
        article_data {dict} -- Article data dictionary. Needs keys url and content.
        openai_api_key {str} -- API key for the OpenAI API.

    Returns:
        dict -- Patched article_data with the summary under the bullet_point_summary key.
    """
    tic = time.time()

    logging.debug(
        "Summarizing article",
        {
            "url": article_data["url"],
            "len": len(article_data["content"]),
        },
    )

    # Splitting the article to accommodate the model's context window, and creating langchain Documents
    split_article = text_splitter.split_text(article_data["content"])
    split_langchain_docs = [Document(page_content=i) for i in split_article]

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
    summary_resp = summarize_chain(
        inputs={"input_documents": split_langchain_docs, "title": article_data["title"]},
        return_only_outputs=False,
    )

    # Patch article_data with the summary and return
    article_data["summary"] = summary_resp["output_text"]
    logging.debug(
        "Successfully created summary of the article",
        {
            "url": article_data["url"],
            "summary_len": len(article_data["summary"]),
            "time": time.time() - tic,
        },
    )
    print(article_data)
    return article_data
