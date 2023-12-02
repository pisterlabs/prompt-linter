"""
Summarization of biomedical papers. Exploratory evaluation using GPT-3.

Author: Michael A. Hedderich
Last update: 22-01-30
"""
import os
import openai

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

def perform_gpt_call(prompt):
    openai.api_key = OPENAI_API_KEY
    response = openai.Completion.create(
        engine="text-davinci-002",
        prompt=prompt,
        temperature=0.5,
        max_tokens=256,
        top_p=1.0,
        frequency_penalty=0.0,
        presence_penalty=0.0
    )
    return response['choices'][0]['text']

def read_and_filter_pdf_extraction(path, minimum_line_length = 10):
    with open(path, "r") as in_file:
        lines = in_file.readlines()
        lines = [line.strip() for line in lines if len(line) >= minimum_line_length]
        text = " ".join(lines)
    return text

def shorten_text_for_prompt(text):
    """ GPT has a maximum token length of 4097,
    so we need to throw away a lot of tokens.
    This is a first approach, could be improved.
    TODO: Revisit
    """

    # remove acknowledgments and references
    reference_start = text.find("References")
    acknowledgments_start = text.find("Acknowledgments")
    if reference_start == -1 or acknowledgments_start == -1:
        text_end = max(acknowledgments_start, reference_start)
    else:
        text_end = min(acknowledgments_start, reference_start)
    print(f"Removing everything after character #{text_end} as these are probably only the acknowledgments "
          f"or references.")
    text = text[:text_end]

    # remove the text before the start of the actual paper
    # find either Abstract or Introduction and use the earlier one
    abstract_start = text.find("Abstract")
    introduction_start = text.find("Introduction")
    if abstract_start == -1 or introduction_start == -1:
        text_start = max(abstract_start, introduction_start)
        if text_start == -1:
            text_start = 0
    else:
        text_start = min(abstract_start, introduction_start)
    print(f"Removing everything before character #{text_start} as these come before the abstract or introduction.")
    text = text[text_start:]

    # remove hyperlinks
    text = " ".join([token for token in text.split(" ") if not token.startswith("http")])

    # remove part of the text
    # TODO: Definitely revisit. Do something more meaningful and less influenced by random sampling
    # OpenAI estimates that one token is 0.75 words
    # (https://help.openai.com/en/articles/4936856-what-are-tokens-and-how-to-count-them)
    # We thus can have for a max context length of 4097 tokens (-256 for completion), a maximum of 3841 words.
    # Remove the middle part of the paper that might have the least amount of relevant information for the summary.
    words = text.split(" ")
    count_to_remove = len(words) - 2500
    if count_to_remove > 0:
        middle_of_text = int(len(words) / 2)
        half_count_to_remove = int(count_to_remove / 2)
        words = words[:(middle_of_text-half_count_to_remove)] + words[(middle_of_text+half_count_to_remove):]
        text = " ".join(words)

    return text

def summarize_with_gpt(path):
    text = read_and_filter_pdf_extraction(path)
    text = shorten_text_for_prompt(text)
    prompt = f'Summarize this: "{text}"'
    print(prompt)
    response = perform_gpt_call(prompt)
    print("Done")
    print(response)

if __name__ == "__main__":
    summarize_with_gpt("../data/1.38037-PB1-9531-R2.pdf.tika.content.txt")