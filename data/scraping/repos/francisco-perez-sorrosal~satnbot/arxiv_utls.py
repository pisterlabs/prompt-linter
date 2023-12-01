import arxiv
from langchain import OpenAI, PromptTemplate
from langchain.chains.summarize import load_summarize_chain
from langchain.document_loaders import OnlinePDFLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

from utils import get_gai_completion


def download_pdf_from_arxiv(arxiv_id: str, filename="downloaded-paper.pdf"):
    paper = next(arxiv.Search(id_list=[arxiv_id]).results())
    dest_file = paper.download_pdf(filename=filename)
    return paper, dest_file


def extract_text_from_arxiv_pdf(arxiv_id, chars_per_chunk, overlap_chars):
    paper, dest_file = download_pdf_from_arxiv(arxiv_id)
    loader = PyPDFLoader(dest_file)
    test_splitter = RecursiveCharacterTextSplitter(  # Set a really small chunk size, just to show.
        chunk_size=chars_per_chunk, chunk_overlap=overlap_chars, length_function=len
    )
    return paper.title, loader.load_and_split(text_splitter=test_splitter)


def summarize_arxiv_paper(text, style, style_items, language):
    if style == "paragraph":
        summary_style = f"{style_items} paragraph"
    elif style == "bulletpoints":
        summary_style = f"{style_items} bullet points"
    elif style == "sonnet":
        summary_style = "a sonnet style"
    else:
        summary_style = "a single sentence"

    prompt = f""""
        Summarize the technical text, delimited by triple
        backticks, in {language} in {summary_style}:
        ```{text}```
        """

    return get_gai_completion(prompt)


def summarize_arxiv_paper_lc(docs, style, style_items, language):
    if style == "paragraph":
        summary_style = f"{style_items} paragraph"
    elif style == "bulletpoints":
        summary_style = f"{style_items} bullet points"
    elif style == "sonnet":
        summary_style = "a sonnet style"
    else:
        summary_style = "a single sentence"

    prompt_template = """
        Summarize the technical text, delimited by triple
        =, in {language} in {summary_style}:
        ==={text}===
    """

    PROMPT = PromptTemplate(template=prompt_template, input_variables=["text", "language", "summary_style"])
    chain = load_summarize_chain(
        OpenAI(temperature=0),
        chain_type="map_reduce",
        return_intermediate_steps=True,
        map_prompt=PROMPT,
        combine_prompt=PROMPT,
    )
    output = chain(
        {"input_documents": docs, "language": language, "summary_style": summary_style}, return_only_outputs=True
    )
    print(output)
    return output["output_text"]
