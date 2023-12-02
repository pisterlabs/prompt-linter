"""
This is a module that runs bloggpt as a recurrent RAG (retrieval augmented generation) chain. It utilizes pinecone as a 
vector database to store and retrieve documents. It also uses the openai API to generate the blog.
"""
import os
import time
from pprint import pprint

import pinecone
from dotenv import load_dotenv
from langchain.callbacks import get_openai_callback
from langchain.chains import LLMChain, RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.text_splitter import TokenTextSplitter
from langchain.vectorstores import Pinecone

from prompts.prompts import (
    OUTLINE_PROMPT,
    RECURRENT_RQNA_SYSTEM_PROMPT,
    REWRITE_PROMPT,
    TOPIC_PROMPT,
)
from utils.main_utils import bprint, generate_final_blog, gprint, rprint
from utils.web_utils import search_and_extract_web_url

# Load environment variables from .env file
load_dotenv()

# Use the environment variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENV")


def save_doc_list_to_file(lst, filename):
    """
    This function takes a list of strings and a filename, and writes each string to a new line in the file.

    Parameters:
    lst (list): The list of strings to write to the file.
    filename (str): The name of the file to write to.

    Returns:
    None
    """
    with open(filename, "w") as f:
        for item in lst:
            f.write("%s\n" % item[0].page_content)


def split_outline_prompt(OUTLINE_PROMPT):
    """
    This function splits the OUTLINE_PROMPT into headers and blog sections.

    Parameters:
    OUTLINE_PROMPT (str): The outline prompt for the blog.

    Returns:
    headers (list): The headers of the blog sections.
    blog_sections (list): The blog sections.
    """
    headers = []
    for text in OUTLINE_PROMPT.split("\n"):
        if text.startswith("#"):
            headers.append(text.split("#")[1].strip())
    print("headers: ", headers)

    blog_sections = OUTLINE_PROMPT.split("\n\n")

    return headers, blog_sections


def generate_blog_section(blog_section, namespace, docsearch, draft_llm_chain, topic):
    """
    This function generates a blog post based on the given blog section and namespace.

    Parameters:
    blog_section (str): The section of the blog to generate.
    namespace (str): The namespace to use for the generation.
    docsearch (Pinecone): The Pinecone instance to use for document search.
    draft_llm_chain (LLMChain): The LLMChain instance to use for draft generation.
    topic (str): The topic of the blog post.

    Returns:
    tuple: The draft output and the retrieved documents.
    """
    docs = docsearch.similarity_search_with_score(topic, k=10, namespace=namespace)
    print("docs: ", docs)
    full_context = "\n".join([doc[0].page_content for doc in docs if doc[1] > 0.85])
    inputs = {"context": full_context, "topic": topic, "blog_section": blog_section}

    with get_openai_callback() as cb:
        draft_llm_output = draft_llm_chain(inputs)

    pprint(draft_llm_output)
    pprint(cb)
    return draft_llm_output, docs


def setup_pinecone_index(index_name):
    """
    This function sets up the Pinecone instance and handles the creation of a new index if necessary.

    Parameters:
    index_name (str): The name of the index.
    embeddings (OpenAIEmbeddings): The OpenAIEmbeddings instance to use.

    Returns:
    Pinecone: The Pinecone instance.
    """
    if index_name in pinecone.list_indexes():
        rprint("Creating New Index")

        # Delete existing index
        pinecone.delete_index("bloggpt")

        # Create fresh index
        pinecone.create_index(
            index_name,
            dimension=1536,
            metric="cosine",
            pods=1,
        )

    # Wait for the index to become active
    while True:
        # Fetch index info
        index_description = pinecone.describe_index(index_name)

        # Check if the index is Ready
        if index_description.status["state"] == "Ready":
            bprint(f"The index {index_name} is active and ready.")
            break
        else:
            bprint(
                f"The index {index_name} is not active. Current state: {index_description.status['state']}"
            )
            bprint("Waiting for the index to become active...")

        # Wait for a while before checking again
        time.sleep(5)  # wait for 10 seconds

    return


def combine_drafts(directory):
    """
    This function combines all the markdown generated blogs into one file.

    Parameters:
    directory (str): The directory where the drafts are stored.

    Returns:
    entire_draft (str): The entire draft of the blog.
    """
    # Define the name of the output file
    output_file = directory + "complete_draft.md"

    # Get a list of all draft files
    draft_files = [
        f for f in os.listdir(directory) if f.startswith("draft_") and f.endswith(".md")
    ]

    # Sort the files by their number to maintain the order
    draft_files.sort(key=lambda x: int(x.split("_")[1].split(".")[0]))

    # Initialize an empty string to store the entire draft
    entire_draft = ""

    # Open the output file in write mode
    with open(output_file, "w") as outfile:
        # Iterate over each draft file
        for filename in draft_files:
            # Open the draft file in read mode
            with open(directory + filename, "r") as infile:
                # Read the contents of the draft file
                contents = infile.read()

                # Write the contents of the draft file to the output file
                outfile.write(contents)

                # Add the contents of the draft file to the entire draft string
                entire_draft += contents

                # Add a newline to separate the contents of different draft files
                outfile.write("\n\n")
                entire_draft += "\n\n"

    gprint(f"All drafts have been combined into {output_file}.")
    return entire_draft


def main():
    """
    The main function to run the script.

    Returns:
    None
    """
    topic = TOPIC_PROMPT.split(":")[1].strip()
    print("topic: ", topic)

    headers, blog_sections = split_outline_prompt(OUTLINE_PROMPT)

    # initialize pinecone
    pinecone.init(
        api_key=PINECONE_API_KEY,  # find at app.pinecone.io
        environment=PINECONE_ENV,  # next to api keySorry for the abrupt cut-off. Here's the continuation of the refactored code:
    )

    rprint("Creating VectorDB")
    index_name = "bloggpt"
    embeddings = OpenAIEmbeddings()

    setup_pinecone_index(index_name)

    num_generated = 0
    for header, blog_section in zip(headers, blog_sections):
        rprint(f"Searching Google for {topic}, {header}")
        search_and_extract_web_url(f"{topic}, {header}")

        with open("outputs/web_search_texts.txt") as f:
            web_search_texts: str = f.read()

        # Remove special tokenizer tokens from text
        web_search_texts = web_search_texts.replace("<|endoftext|>", " ")

        text_splitter = TokenTextSplitter(chunk_size=300, chunk_overlap=100)

        # # This is a hack to get around the fact that the tokenizer doesn't like special tokens
        # text_splitter._disallowed_special = (
        #     text_splitter._tokenizer.special_tokens_set - {"<|endoftext|>"}
        # )

        texts = text_splitter.split_text(web_search_texts)
        docsearch = Pinecone.from_texts(
            texts, embeddings, index_name=index_name, namespace=header, batch_size=16
        )

        blog_section_rqna_prompt = PromptTemplate(
            template=RECURRENT_RQNA_SYSTEM_PROMPT,
            input_variables=["context", "topic", "blog_section"],
        )

        draft_llm = ChatOpenAI(
            model_name="gpt-4-0613",
            openai_api_key=OPENAI_API_KEY,
        )

        draft_llm_chain = LLMChain(
            llm=draft_llm,
            prompt=blog_section_rqna_prompt,
            verbose=True,
        )

        # Generate the blog with the agent
        rprint("Generating First Draft")
        draft_llm_output, retrieved_docs = generate_blog_section(
            blog_section, header, docsearch, draft_llm_chain, topic
        )
        print(draft_llm_output["text"])

        generated_blog = draft_llm_output["text"]

        save_doc_list_to_file(
            retrieved_docs, f"outputs/retrieved_docs_{num_generated}.txt"
        )

        # Save the blog in a markdown file
        with open(f"outputs/draft_{num_generated}.md", "w") as f:
            f.write(generated_blog)

        num_generated += 1

    # Combine the markdown generated blogs into one
    rprint("Combining the markdown generated blogs into one")
    entire_draft = combine_drafts("outputs/")

    # Refine the generated blog
    rprint("Generating Final Blog")
    generate_final_blog(entire_draft, TOPIC_PROMPT, OPENAI_API_KEY)


if __name__ == "__main__":
    main()
