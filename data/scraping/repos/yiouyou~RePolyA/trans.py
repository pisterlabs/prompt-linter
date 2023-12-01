from repolya._log import logger_paper

from langchain.schema import Document
from langchain.document_transformers import DoctranTextTranslator


async def trans_to(_en, _lang):
    logger_paper.info(_en)
    _docs = [Document(page_content=_en)]
    _translator = DoctranTextTranslator(language=_lang)
    _translated_docs = await _translator.atransform_documents(_docs)
    _zh = _translated_docs[0].page_content
    logger_paper.info(_zh)
    return _zh

