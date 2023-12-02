import concurrent.futures
import logging
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional, cast, NamedTuple

import semantic_kernel as sk
from pydantic import Field
from semantic_kernel.connectors.ai.open_ai.services.open_ai_chat_completion import OpenAIChatCompletion

logger = logging.getLogger(__name__)



@dataclass
class Document():
    page_content: str
    metadata: dict = Field(default_factory=dict)

# Borrowed from Langchain TextLoader
# Below code is borrowed from Langchain's TextLoader implementation
class FileEncoding(NamedTuple):
    """File encoding as the NamedTuple."""

    encoding: Optional[str]
    """The encoding of the file."""
    confidence: float
    """The confidence of the encoding."""
    language: Optional[str]
    """The language of the file."""


class ContentLoader(ABC):
    def __init__(self):
        self.module = "Loader"

    @abstractmethod
    def load(self) -> List[Document]:
        """Implement it into the derived loader class."""

    def detect_file_encodings(file_path: str, timeout: int = 5) -> List[FileEncoding]:
        """Try to detect the file encoding.

        Returns a list of `FileEncoding` tuples with the detected encodings ordered
        by confidence.

        Args:
            file_path: The path to the file to detect the encoding for.
            timeout: The timeout in seconds for the encoding detection.
        """
        import chardet

        def read_and_detect(file_path: str) -> List[dict]:
            with open(file_path, "rb") as f:
                rawdata = f.read()
            return cast(List[dict], chardet.detect_all(rawdata))

        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(read_and_detect, file_path)
            try:
                encodings = future.result(timeout=timeout)
            except concurrent.futures.TimeoutError:
                raise TimeoutError(
                    f"Timeout reached while detecting encoding for {file_path}"
                )

        if all(encoding["encoding"] is None for encoding in encodings):
            raise RuntimeError(f"Could not detect encoding for {file_path}")
        return [FileEncoding(**enc) for enc in encodings if enc["encoding"] is not None]


class TextContentLoader(ContentLoader):
    def __init__(self, content_path: str,
                 content_encoding: Optional[str] = None,
                 autodetect_content_encoding: bool = False):
        """Initialize the content details."""
        super().__init__()
        self.content_path = content_path
        self.content_encoding = content_encoding
        self.autodetect_content_encoding = autodetect_content_encoding

    def load(self) -> List[Document]:
        """Load from file path."""
        text = ""
        try:
            with open(self.content_path, encoding=self.content_encoding) as f:
                text = f.read()
        except UnicodeDecodeError as e:
            if self.autodetect_content_encoding:
                detected_encodings = self.detect_file_encodings(self.content_path)
                for encoding in detected_encodings:
                    logger.debug(f"Trying encoding: {encoding.encoding}")
                    try:
                        with open(self.content_path, encoding=encoding.encoding) as f:
                            text = f.read()
                        break
                    except UnicodeDecodeError:
                        continue
            else:
                raise RuntimeError(f"Error loading {self.content_path}") from e
        except Exception as e:
            raise RuntimeError(f"Error loading {self.content_path}") from e

        metadata = {"source": self.content_path}
        return [Document(page_content=text, metadata=metadata)]
# Borrowed from Langchain TextLoader

if __name__=="__main__":

    def get_files_from_dir(dir):
        files = [os.path.join(dir, f) for f in os.listdir(dir) if os.path.isfile(os.path.join(dir, f))]

        return files

    kernel = sk.Kernel()
    api_key, org_id = sk.openai_settings_from_dot_env()
    kernel.add_text_completion_service("gpt-3.5-turbo-16k", OpenAIChatCompletion("gpt-3.5-turbo-16k", api_key))

    list_all_docs = []
    files = get_files_from_dir("/Users/joyeed/semantic_kernel/semantic_kernel_examples/data/pdf/chunks/")

    for file in files:
        tl = TextContentLoader(file)
        documents = tl.load()
        documents[0].metadata["source"] = file
        list_all_docs.append(documents[0])

    map_prompt = """
    Generate a concise and coherent summary from the given document. 
    Condense the document content into a well-written summary that captures the main ideas, key points, and insights presented in the document. 
    Prioritize clarity and brevity while retaining the essential information. 
    Aim to convey the document's core message and any supporting details that contribute to a comprehensive understanding. 
    Craft the summary to be self-contained, ensuring that readers can grasp the content even if they haven't read the document. 
    The goal is to create a summary that effectively communicates the document's content while being easily digestible and engaging.
    Summary should NOT be more than 150 words.
    
    Document:
    {{$document}}
    """

    map_chain  = kernel.create_semantic_function(
        prompt_template=map_prompt,
        description="Extracts main themes from a set of documents",
        max_tokens=2000
    )

    themes =[]
    for document in list_all_docs:
        sk_context = kernel.create_new_context()
        sk_context["document"] = document.page_content
        answer = map_chain.invoke(context=sk_context)
        themes.append(str(answer))

    reduce_prompt = """
    Generate a concise and coherent summary from the given document summaries. 
    Condense the document summaries into a well-written consolidated summary that captures the main ideas, key points, and insights presented in the document summaries. 
    Prioritize clarity and brevity while retaining the essential information. 
    Aim to convey the document's core message and any supporting details that contribute to a comprehensive understanding. 
    Craft the summary to be self-contained, ensuring that readers can grasp the content even if they haven't read the document. 
    The goal is to create a summary that effectively communicates the content from all the summaries, while being easily digestible and engaging.
    Final summary should NOT be more than 1000 words.

    Document:
    {{$document_summaries}}
    """

    reduce_chain = kernel.create_semantic_function(
        prompt_template=reduce_prompt,
        description="creates a final summary from a set of document summaries",
        max_tokens=2000
    )
    sk_context = kernel.create_new_context()
    sk_context["document_summaries"] = "\n".join([t for t in themes])
    answer = reduce_chain.invoke(context=sk_context)

    print(answer)

