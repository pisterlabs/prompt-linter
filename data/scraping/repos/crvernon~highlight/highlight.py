
import openai
import PyPDF2
import tiktoken
from tqdm import tqdm


def get_token_count(text, model="gpt-4"):
    """Get the token count of text content based on the model"""

    encoding = tiktoken.encoding_for_model(model)
    encoded_text = encoding.encode(text)
    n_text_tokens = len(encoded_text)

    return n_text_tokens


def read_pdf(file_object: object,
             reference_indicator: str = "References\n") -> dict:
    """Strip content from PDF file to text.  Stop collecting after the
    reference page has been hit

    """

    content = ""
    n_pages = 0

    # creating a pdf reader object
    reader = PyPDF2.PdfReader(file_object)

    for page in reader.pages:

        page_content = page.extract_text()

        if reference_indicator in page_content:
            content += page_content
            break

        else:
            content += page_content

        n_pages += 1

    content = content.split(reference_indicator)[0]

    return {
        "content": content,
        "n_pages": n_pages,
        "n_characters": len(content),
        "n_words": len(content.split(" ")),
        "n_tokens": get_token_count(content)
    }


def read_text(file_object: object,) -> dict:
    """Read text file input."""

    content = bytes.decode(file_object.read(), 'utf-8')

    return {
        "content": content,
        "n_pages": 1,
        "n_characters": len(content),
        "n_words": len(content.replace("\n", " ").split()),
        "n_tokens": get_token_count(content)
    }


def content_reduction(document_list,
                      system_scope):
    """Remove irrelevant content from input text."""

    prompt = """Remove irrelevant content from the following text.\n\n{text}\n\n}"""

    content = ""
    for i in tqdm(range(len(document_list))):
        page_content = document_list[i].page_content
        page_tokens = get_token_count(page_content)

        messages = [{"role": "system",
                     "content": system_scope},
                    {"role": "user",
                     "content": prompt.format(text=page_content)}]

        response = openai.ChatCompletion.create(
            model="gpt-4",
            max_tokens=page_tokens,
            temperature=0.0,
            messages=messages)

        content += response["choices"][0]["message"]["content"]

    return content


def generate_content(system_scope,
                     prompt,
                     max_tokens=50,
                     temperature=0.0,
                     max_allowable_tokens=8192):

    n_prompt_tokens = get_token_count(prompt) + max_tokens

    if n_prompt_tokens > max_allowable_tokens:
        raise RuntimeError(
            f"ERROR:  input text tokens needs to be reduced due to exceeding the maximum allowable tokens per prompt by {n_prompt_tokens - max_allowable_tokens} tokens.")

    messages = [{"role": "system",
                 "content": system_scope},
                {"role": "user",
                 "content": prompt}]

    response = openai.ChatCompletion.create(
        model="gpt-4",
        max_tokens=max_tokens,
        temperature=temperature,
        messages=messages)

    content = response["choices"][0]["message"]["content"]

    return content


