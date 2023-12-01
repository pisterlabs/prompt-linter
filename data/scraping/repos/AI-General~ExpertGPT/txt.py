from langchain.document_loaders import TextLoader
from models.files import File

from .common import process_file


async def process_txt(file: File, enable_summarization, brain_id):
    return await process_file(
        file=file,
        loader_class=TextLoader,
        enable_summarization=enable_summarization,
        brain_id=brain_id
    )
