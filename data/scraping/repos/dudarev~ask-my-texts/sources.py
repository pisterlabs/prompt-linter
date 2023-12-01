from pathlib import Path
from pprint import pprint

import frontmatter
from langchain.text_splitter import CharacterTextSplitter


CHUNK_SIZE = 512
CHUNK_OVERLAP = 128

# texts directory relative to this file directory
TEXTS_DIR = Path(__file__).parent.parent / "texts"
if not TEXTS_DIR.exists():
    raise Exception(
        "Please put your texts to be searched in a folder called 'texts' in the root of the project. "
        "See README.md for more details."
    )

SOURCES_DATA_DIR = Path(__file__).parent.parent / "data/sources"
SOURCES_DATA_DIR.mkdir(exist_ok=True, parents=True)
SOURCES_DATA_FILE = SOURCES_DATA_DIR / "sources.json"


def iter_sources_and_chunks():
    for f in TEXTS_DIR.iterdir():
        with open(f) as fp:
            fm = frontmatter.load(fp)
            if fm.get("title"):
                text_splitter = CharacterTextSplitter(
                    chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
                )
                file_chunks = text_splitter.split_text(fm.content)
                file_sources = [f"{f.name}-{i}" for i in range(len(file_chunks))]
                for source, chunk in zip(file_sources, file_chunks):
                    yield source, chunk


def get_sources_dict():
    return {source: chunk for source, chunk in iter_sources_and_chunks()}


def get_source_filepath(source: str) -> Path:
    """
    For source that looks like "filename-chunkid", return the path to the file.
    """
    filename, _ = source.rsplit("-", 1)
    return TEXTS_DIR / filename


def get_metadata_for_source(source: str):
    source_filepath = get_source_filepath(source)
    with open(source_filepath) as fp:
        fm = frontmatter.load(fp)
    return fm.metadata


def get_sources_markdown(sources_str: str):
    sources_list = map(lambda x: x.strip(), sources_str.split(","))
    all_sources_dict = get_sources_dict()
    markdown = ""
    for source in sources_list:
        if source in all_sources_dict:
            metadata = get_metadata_for_source(source)
            markdown += (
                f"## [{metadata['title']}]({metadata['url']})\n\n"
                f"{all_sources_dict[source]}\n\n"
            )
    return markdown


if __name__ == "__main__":
    for source in get_sources_dict().keys():
        print(source)
        print(get_source_filepath(source))
        pprint(get_metadata_for_source(source))
