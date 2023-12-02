from langchain.docstore.document import Document

# copied and modified from the unstructured library: https://github.com/Unstructured-IO/unstructured/blob/main/unstructured/partition/xlsx.py

from tempfile import SpooledTemporaryFile
from typing import IO, BinaryIO, Optional, Union, cast

import pandas as pd

from unstructured.documents.elements import (
    Element,
    ElementMetadata,
    Table,
    process_metadata,
)
from unstructured.file_utils.filetype import FileType, add_metadata_with_filetype
from unstructured.partition.common import exactly_one, spooled_to_bytes_io_if_needed


@process_metadata()
@add_metadata_with_filetype(FileType.XLSX)
def partition_xlsx_m(
    filename: Optional[str] = None,
    file: Optional[Union[IO, SpooledTemporaryFile]] = None,
    metadata_filename: Optional[str] = None,
    include_metadata: bool = True,
    **kwargs,
) -> list[Element]:
    """Partitions Microsoft Excel Documents in .xlsx format into its document elements.
    Parameters
    ----------
    filename
        A string defining the target filename path.
    file
        A file-like object using "rb" mode --> open(filename, "rb").
    metadata_filename
        The filename to use for the metadata.
    include_metadata
        Determines whether or not metadata is included in the output.
    """
    exactly_one(filename=filename, file=file)

    if filename:
        sheets = pd.read_excel(filename, sheet_name=None)
    else:
        f = spooled_to_bytes_io_if_needed(
            cast(Union[BinaryIO, SpooledTemporaryFile], file))
        sheets = pd.read_excel(f, sheet_name=None)

    metadata_filename = filename or metadata_filename

    elements: list[Element] = []
    page_number = 0
    for sheet_name, table in sheets.items():
        page_number += 1
        text = table.to_csv(index=False, header=False, na_rep="", sep=",")

        if include_metadata:
            metadata = ElementMetadata(
                page_name=sheet_name,
                page_number=page_number,
                filename=metadata_filename,
            )
        else:
            metadata = ElementMetadata()

        table = Table(text=text, metadata=metadata)
        elements.append(table)

    return elements


def load_xlsx_csv(filePath: str) -> list[Document]:
    """loads the excel file, stores each sheet of the document as a langchain
    Document with the text in CSV format.
    """
    elements = partition_xlsx_m(filePath)
    docs = []
    for element in elements:
        metadata = {"source": filePath}
        if hasattr(element, "metadata"):
            metadata.update(element.metadata.to_dict())
        if hasattr(element, "category"):
            metadata["category"] = element.category
        metadata["text_format"] = "text/csv"
        docs.append(Document(page_content=str(element), metadata=metadata))
    return docs


def load_xlsx(filePath: str) -> list[Document]:
    return load_xlsx_csv(filePath)
