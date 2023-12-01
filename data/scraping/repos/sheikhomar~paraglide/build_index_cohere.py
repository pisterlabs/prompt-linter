import os
from pathlib import Path
from typing import List

import click
import pandas as pd
from anyio import run as anyio_run
from llama_index import ServiceContext, VectorStoreIndex
from llama_index.embeddings.cohereai import CohereEmbedding
from llama_index.node_parser import SimpleNodeParser
from llama_index.schema import (
    BaseNode,
    Document,
    MetadataMode,
    NodeRelationship,
    RelatedNodeInfo,
)
from nltk.tokenize import word_tokenize
from paraglide.data.models import Statute, StructuredText, TextType

# logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
# logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))


class StatuteDocumentBuilder:
    def __init__(self, statute: Statute) -> None:
        self._statute = statute

    def build(self) -> List[Document]:
        """Load the data from the Statute model into a list of Documents."""

        documents: List[Document] = []

        text_template = "Meta data:\n{metadata_str}\n\nIndhold:\n{content}"
        metadata_template = "{key}: {value},"
        metadata_seperator = " "

        for chapter in self._statute.chapters:
            for paragraph in chapter.paragraphs:
                paragraph_content = self._concat_texts(paragraph.texts)
                # paragraph_header_path = f"{self._statute.title} > {chapter.title} > {paragraph.reference}"
                paragraph_doc = Document(
                    text=paragraph_content,
                    metadata={
                        "GUID": paragraph.guid,
                        "Type": "Paragraf",
                        "Reference": paragraph.reference,
                        "Kapitel nummer": chapter.number,
                        "Kapitel overskrift": chapter.title,
                    },
                    excluded_embed_metadata_keys=["GUID"],
                    excluded_llm_metadata_keys=["GUID"],
                    text_template=text_template,
                    metadata_template=metadata_template,
                    metadata_seperator=metadata_seperator,
                )
                documents.append(paragraph_doc)

                for section in paragraph.sections:
                    section_content = self._concat_texts(section.texts)
                    # section_header_path = f"{paragraph_header_path} > {section.reference}"
                    documents.append(
                        Document(
                            text=section_content,
                            metadata={
                                "GUID": section.guid,
                                "Reference": section.reference,
                                "Kapitel nummer": chapter.number,
                                "Kapitel overskrift": chapter.title,
                            },
                            excluded_embed_metadata_keys=["GUID"],
                            excluded_llm_metadata_keys=["GUID"],
                            text_template=text_template,
                            metadata_template=metadata_template,
                            metadata_seperator=metadata_seperator,
                            relationships={
                                NodeRelationship.PARENT: RelatedNodeInfo(
                                    node_id=paragraph_doc.id_,
                                )
                            },
                        )
                    )

        return documents

    def _concat_texts(self, texts: List[StructuredText]) -> str:
        """Concatenate the paragraph content with the surrounding context."""
        final_text = ""
        for text in texts:
            if text.type == TextType.plain:
                final_text += f"{text.text}\n"
            elif text.type == TextType.list:
                final_text += f"{text.reference} {text.text}\n"
            else:
                raise ValueError(f"Unknown text type: {text.type}")
        return final_text


async def convert_statute_to_docs(statute_path: Path) -> List[Document]:
    """Convert a statute from a file to a list of `Document` objects.

    Args:
        statute_path (Path): path to the statute file.

    Returns:
        List[Document]: list of `Document` objects that can be used
        by LlamaIndex.
    """
    print(f"Loading statute from {statute_path}")
    statute = Statute.from_json_file(statute_path=statute_path)

    print(f"Creating documents from statute '{statute.title}'")
    docs = StatuteDocumentBuilder(statute=statute).build()

    return docs


async def build_index(
    index_dir: Path,
    statute_path: Path,
    service_context: ServiceContext,
) -> VectorStoreIndex:
    """Build a LlamaIndex from a statute file."""

    print("Converting statute to documents.")
    docs = await convert_statute_to_docs(statute_path=statute_path)

    print(f"Creating nodes from {len(docs)} documents...")
    nodes = service_context.node_parser.get_nodes_from_documents(
        documents=docs,
        show_progress=True,
    )
    print(f"Buidling index with {len(nodes)} nodes.")
    index = VectorStoreIndex(
        service_context=service_context,
        nodes=nodes,
        show_progress=True,
    )

    print(f"Persisting index to {index_dir}")
    index.storage_context.persist(persist_dir=index_dir)

    return index


async def print_node_stats(nodes: List[BaseNode]) -> None:
    embed_texts = [node.get_content(metadata_mode=MetadataMode.EMBED) for node in nodes]

    embed_text_lens = [len(text) for text in embed_texts]

    word_counts = [len(word_tokenize(text)) for text in embed_texts]

    df_data = pd.DataFrame(
        data={
            "char_len": embed_text_lens,
            "word_count": word_counts,
            "text": embed_texts,
        }
    )

    print(df_data.describe())


async def runner(statute_path: Path, cohere_api_key: str, index_dir: Path) -> None:
    embed_model = CohereEmbedding(
        cohere_api_key=cohere_api_key,
        model_name="embed-multilingual-v3.0",
        input_type="search_document",
    )

    node_parser = SimpleNodeParser.from_defaults(
        chunk_size=512,
        chunk_overlap=10,
    )

    service_context = ServiceContext.from_defaults(
        llm=None,
        embed_model=embed_model,
        node_parser=node_parser,
    )

    await build_index(
        index_dir=index_dir,
        statute_path=statute_path,
        service_context=service_context,
    )


@click.command()
@click.option(
    "--statute-path",
    default="data/eli-lta-2023-1180.json",
    help="Path to the statute json file.",
)
@click.option(
    "--cohere-api-key",
    default=None,
    help="Cohere API key. Defaults to COHERE_API_KEY environment variable.",
)
@click.option(
    "--index-dir",
    default="data/llama-indices/cohere-embed-v3",
    help="Path to the directory where the index should be stored.",
)
def main(statute_path: str, cohere_api_key: str, index_dir: str) -> None:
    statute_path_ = Path(statute_path)
    if not statute_path_.exists():
        raise click.BadParameter(f"File {statute_path_} does not exist.")

    index_dir_ = Path(index_dir)
    if index_dir_.exists():
        raise click.BadParameter(
            f"Path {index_dir_} already exists. To rebuild the index you need to remove it."
        )

    if cohere_api_key is None or len(cohere_api_key) == 0:
        cohere_api_key = os.environ["COHERE_API_KEY"]

    if cohere_api_key is None or len(cohere_api_key) == 0:
        raise click.BadParameter(
            message=(
                "Missing the Cohere API key. Either set COHERE_API_KEY "
                "environment variable or pass in the key via the "
                "--cohere-api-key option."
            )
        )

    anyio_run(runner, statute_path_, cohere_api_key, index_dir_)


if __name__ == "__main__":
    main()
