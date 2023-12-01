from langchain.chains import RetrievalQAWithSourcesChain
import chainlit as cl

from chain_factory import load_all_chains
from geolocation import extract_ip_address, geolocate
from source_splitter import source_splitter
from chainlit.context import get_emitter

from log_init import logger

from pathlib import Path
from typing import Dict, Optional

from config import cfg

KEY_META_DATAS = "metadatas"
KEY_TEXTS = "texts"
KEY_GEOLOCATION_COUNTRY_CODE = "geolocation_country_code"


def set_session_vars(user_session_dict: Dict):
    for k, v in user_session_dict.items():
        cl.user_session.set(k, v)


def create_pdf(pdf_name: str, pdf_path: str) -> Optional[cl.File]:
    """
    Creates a file download button for a PDF file in case it is found.

    Parameters:
    pdf_name (str): The file name
    pdf_path (str): The file name

    Returns:
    RetrievalQAWithSourcesChain: The QA chain
    """
    logger.info(f"Creating pdf for {pdf_path}")
    # Sending a pdf with the local file path
    country_code = cl.user_session.get(KEY_GEOLOCATION_COUNTRY_CODE)
    country_config = cfg.location_persistence_map.get(country_code)
    if country_config:
        logger.info("country_config found")
        doc_location: Path = country_config.get("doc_location")
        doc_path = doc_location / pdf_path
        if doc_path.exists():
            logger.info("Creating pdf component")
            return cl.File(
                name=pdf_name, display="inline", path=str(doc_path.absolute())
            )
        else:
            logger.info(f"doc path {doc_path} does not exist.")
    return None


@cl.langchain_factory(use_async=True)
async def init():
    """
    Loads the vector data store object and the PDF documents. Creates the QA chain.
    Sets up some session variables and removes the Chainlit footer.

    Parameters:
    use_async (bool): Determines whether async is to be used or not.

    Returns:
    RetrievalQAWithSourcesChain: The QA chain
    """

    emitter = get_emitter()
    # Please note this works only with a modified version of Streamlit
    # The repo with this modification are here: https://github.com/gilfernandes/chainlit_hr_extension

    country_code = "GB"
    geolocation_failed = False

    try:
        remote_address = extract_ip_address(emitter.session.environ)
        geo_location = geolocate(remote_address)

        if geo_location.country_code != "Not found":
            country_code = geo_location.country_code
            # await display_location_details(geo_location, country_code)
    except:
        logger.exception("Could not locate properly")
        geolocation_failed = True

    if geolocation_failed:
        await cl.Message(content=f"Geolocation failed ... I do not know where you are.").send()
    else:
        logger.info(f"Geo location: {geo_location}")

    msg = cl.Message(content=f"Processing files. Please wait.")
    await msg.send()
    chain_dict = load_all_chains(country_code)
    qa_data = chain_dict[country_code]

    documents = qa_data.documents

    chain: RetrievalQAWithSourcesChain = qa_data.chain
    metadatas = [d.metadata for d in documents]
    texts = [d.page_content for d in documents]

    set_session_vars(
        {
            KEY_META_DATAS: metadatas,
            KEY_TEXTS: texts,
            KEY_GEOLOCATION_COUNTRY_CODE: country_code,
        }
    )

    
    msg.content = f"You can now ask questions about Onepoint HR ({country_code})!"
    await msg.send()

    return chain


async def display_location_details(geo_location, country_code):
    geo_location_msg = cl.Message(
        content=f"""Geo location: 
- country: {geo_location.country_name}
- country code: {country_code}"""
    )
    await geo_location_msg.send()


@cl.langchain_postprocess
async def process_response(res) -> cl.Message:
    """
    Tries to extract the sources and corresponding texts from the sources.

    Parameters:
    res (dict): A dictionary with the answer and sources provided by the LLM via LangChain.

    Returns:
    cl.Message: The message containing the answer and the list of sources with corresponding texts.
    """
    answer = res["answer"]
    sources = res["sources"].strip()
    source_elements = []

    # Get the metadata and texts from the user session
    metadatas = cl.user_session.get(KEY_META_DATAS)
    all_sources = [m["source"] for m in metadatas]
    texts = cl.user_session.get(KEY_TEXTS)

    found_sources = []
    pdf_elements = []
    if sources:
        logger.info(f"sources: {sources}")
        raw_sources, file_sources = source_splitter(sources)
        for i, source in enumerate(raw_sources):
            try:
                source_name = file_sources[i]
                pdf_element = create_pdf(source_name, source_name)
                if pdf_element:
                    pdf_elements.append(pdf_element)
                    logger.info(f"PDF Elements: {pdf_elements}")
                else:
                    logger.warning(f"No pdf element for {source_name}")

                index = all_sources.index(source)
                text = texts[index]
                found_sources.append(source)
                # Create the text element referenced in the message
                logger.info(f"Found text in {source_name}")
                source_elements.append(cl.Text(content=text, name=source_name))
            except ValueError as e:
                logger.error(f"Value error {e}")
                continue
        if found_sources:
            answer += f"\nSources: {', '.join(found_sources)}"
        else:
            answer += f"\n{sources}"

    logger.info(f"PDF Elements: {pdf_elements}")
    await cl.Message(content=answer, elements=source_elements).send()
    await cl.Message(content="PDF Downloads", elements=pdf_elements).send()


if __name__ == "__main__":
    pass
