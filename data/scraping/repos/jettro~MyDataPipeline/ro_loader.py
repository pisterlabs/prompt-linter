import logging

from langchain.vectorstores import VectorStore

from langchainmod import VraagXMLLoader, AntwoordXMLLoader

lc_loader_logging = logging.getLogger("langchain")  # initialize the main logger


def load_content_vacs(vector_store: VectorStore, offset: int = 0, rows: int = 100) -> None:
    """
    Load the content from the following endpoint:
    https://www.rijksoverheid.nl/opendata/vac-s

    More information about possible parameters can be found here:
    https://www.rijksoverheid.nl/opendata/open-data-filteren

    :param rows: Number of rows to obtain, the API specifies 200 as the maximum
    :param offset: Start row, can be used to paginate through all the records
    :param vector_store: Langchain VectorStore to load the data into
    :return: None
    """
    lc_loader_logging.info("Load the content")

    total = rows
    while total >= rows:
        url = f'https://opendata.rijksoverheid.nl/v1/infotypes/faq?rows={rows}&offset={offset}'
        lc_loader_logging.info(url)
        vraag_xml_loader = VraagXMLLoader(file_path=url)
        docs = vraag_xml_loader.load()
        # Fetch details
        for doc in docs:
            detail_url = doc.metadata["dataurl"]
            antwoord = __load_answer(data_url=detail_url)
            doc.metadata["antwoord"] = antwoord.page_content
        total = len(docs)
        lc_loader_logging.info(f"Store the content: offset {offset}, docs {total}")
        vector_store.add_documents(docs)
        offset = offset + rows


def __load_answer(data_url: str):
    antwoord_xml_loader = AntwoordXMLLoader(file_path=data_url)
    antwoorden = antwoord_xml_loader.load()

    # There should be only one antwoord
    if len(antwoorden) == 1:
        return antwoorden[0]
    else:
        lc_loader_logging.warning(f"No answer found, or to many answers. ({len(antwoorden)})")
