import os
import json
import logging
from typing import Literal, Optional, List, TypedDict, Union, Dict

import requests
import anthropic
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import BSHTMLLoader, UnstructuredHTMLLoader
from langchain.embeddings import (
    HuggingFaceInstructEmbeddings,
    FakeEmbeddings,
    OpenAIEmbeddings,
)
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI, HuggingFaceHub, HuggingFacePipeline
from langchain.chat_models import ChatAnthropic, PromptLayerChatOpenAI
from langchain.chains import RetrievalQA, RetrievalQAWithSourcesChain
from langchain.prompts import HumanMessagePromptTemplate, PromptTemplate
from langchain.schema import HumanMessage, AIMessage, BaseMessage

from src.utils import (
    retry,
    list_depth,
    read_indices_list,
    write_indices_list,
)

LLM_CACHE = {}
EMBEDDING_MODEL_CACHE = {}

# import torch
# from transformers import pipeline

# Does not work on Apple silicon
# dolly_pipe = pipeline(model="databricks/dolly-v2-3b", torch_dtype=torch.bfloat16,
#                          trust_remote_code=True, device_map="auto", return_full_text=True)

CHAINS = {"qa": RetrievalQA, "qa_source": RetrievalQAWithSourcesChain}

LLMS = {
    "openai": OpenAI,
    "hf": HuggingFaceHub,
    "hf-local": lambda **kwargs: HuggingFacePipeline.from_model_id(**kwargs),
    "hf-pipe": HuggingFacePipeline,
    "claude": ChatAnthropic,
    "pl-gpt": PromptLayerChatOpenAI,
}

EMBEDDING_MODELS = {
    "instructor_large": HuggingFaceInstructEmbeddings,
    "fake": FakeEmbeddings,
    "openai": OpenAIEmbeddings,
}

MODEL_KWARGS = {"default": {}, "fake": {"size": 1}}

LLM_ARGS = {
    "default": {},
    "tagger": {"temperature": 0.8},
    "reader": {"temperature": 0},
    "stablelm": {
        "model_id": "stabilityai/stablelm-tuned-alpha-3b",
        "task": "text-generation",
        "model_kwargs": {},
    },
    "dolly": {
        "model_id": "databricks/dolly-v2-3b",
        "task": "text-generation",
        "model_kwargs": {},
    },
    # "dolly-pipe": {
    #     "pipeline": dolly_pipe
    # }
}

Song = TypedDict("Song", {"name": str, "artist": str})

MessageChain = List[Union[HumanMessage, AIMessage, BaseMessage]]


def is_message_chain(obj):
    return isinstance(obj, list) and all(
        isinstance(msg, (HumanMessage, AIMessage, BaseMessage)) for msg in obj
    )


class NotIndexedError(Exception):
    """Raised when trying to query before indexing documents."""

    pass


class LLM:
    def __init__(
        self,
        llm_id="pl-gpt",
        llm_kwargs_id="default",
    ):
        self.llm_id = llm_id
        if (llm_id, llm_kwargs_id) in LLM_CACHE:
            self.llm = LLM_CACHE[(llm_id, llm_kwargs_id)]
        else:
            self.llm = LLMS[llm_id](**LLM_ARGS[llm_kwargs_id])
            LLM_CACHE[(llm_id, llm_kwargs_id)] = self.llm

    def generate(self, input: Union[str, MessageChain]) -> str:
        if self.llm_id in ["claude", "pl-gpt"]:
            if not is_message_chain(input):
                input = [HumanMessage(content=input)]
            return self.llm(input).content
        return self.llm(input)

    def generate_with_prompt(
        self,
        template: str,
        vars: Dict[str, str],
        history: Optional[MessageChain] = None,
    ):
        if self.llm_id in ["claude", "pl-gpt"]:
            if history is None:
                history = []
            prompter = HumanMessagePromptTemplate.from_template(template)
            prompt = prompter.format(**vars)
            input = [*history, prompt]
        else:
            prompter = PromptTemplate(
                input_variables=list(vars.keys()), template=template
            )
            input = prompter.format(**vars)
        result = self.generate(input)
        return result


class BaseTagger(LLM):
    PROMPT = """Generate at least 5 tags for the following snippet. Format the output as a JSON list and return only the list.

{snippet}"""

    def __init__(self, llm_id="openai", llm_kwargs_id="default", template: str = None):
        super().__init__(llm_id=llm_id, llm_kwargs_id=llm_kwargs_id)
        if template is None:
            self.template = BaseTagger.PROMPT
        else:
            self.template = template

    def _parse_list(self, output: str, min_length=0):
        output_obj = json.loads(output)
        assert (
            type(output_obj) == list
        ), f"Expected output to be a list, got {output_obj}"
        if min_length:
            assert (
                len(output_obj) >= min_length
            ), f"Expected at least 5 tags, got {len(output_obj)}"
        return output_obj

    def _normalize_tag(self, tag: str):
        # Remove special characters
        tag = "".join([c for c in tag if c.isalnum() or c == " " or c == "&"])

        # Lowercase
        tag = tag.lower()

        # Remove extra whitespaces
        tag = " ".join([word for word in tag.split(" ") if word])

        return tag

    @retry(3, 0.1)
    def __call__(self, snippet: str):
        output = self.generate_with_prompt(self.template, {"snippet": snippet})
        output_obj = self._parse_list(output, min_length=5)
        return [self._normalize_tag(tag) for tag in output_obj]


class CodeTagger(BaseTagger):
    PROMPT = """Generate at least 5 tags the following piece of code enclosed in code tags. Try to include tags specifying language, libraries/packages used, task(s) fulfilled, programming paradigm. Format the output as a JSON list and return only the list.

<code>
{snippet}
</code>"""

    def __init__(
        self,
        llm_id="openai",
        llm_kwargs_id="default",
    ):
        super().__init__(
            llm_id=llm_id, llm_kwargs_id=llm_kwargs_id, template=CodeTagger.PROMPT
        )


class Freader(LLM):
    def __init__(
        self,
        index_name: str,
        faiss_path: str,
        llm_id="pl-gpt",
        llm_kwargs_id="reader",
        model_id="instructor_large",
        model_kwargs_id="default",
        chain_id: str = "qa_source",
        loader_type: Literal["bs", "unstructured"] = "bs",
        chunk_size: int = 200,
        chunk_overlap: int = 50,
        chain_type: str = "stuff",
        retrieval_only=False,
    ):
        super().__init__(llm_id=llm_id, llm_kwargs_id=llm_kwargs_id)
        self.loader_type = loader_type
        self.index_name = index_name
        self.retrieval_only = retrieval_only
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=lambda x: len(x.split()),
        )
        if (model_id, model_kwargs_id) in EMBEDDING_MODEL_CACHE:
            self.model = EMBEDDING_MODEL_CACHE[(model_id, model_kwargs_id)]
        else:
            self.model = EMBEDDING_MODELS[model_id](**MODEL_KWARGS[model_kwargs_id])
            EMBEDDING_MODEL_CACHE[(model_id, model_kwargs_id)] = self.model
        self.faiss_path = faiss_path
        self.chain_type = chain_type
        self.chain_id = chain_id
        if os.path.exists(faiss_path):
            self.db = FAISS.load_local(faiss_path, self.model)
            self._init_chain()

    def _init_chain(self):
        self.chain = CHAINS[self.chain_id].from_chain_type(
            llm=self.llm,
            chain_type=self.chain_type,
            retriever=self.db.as_retriever(),
        )
        self.chain.return_source_documents = True

    def index_raw(self, raw: str, metadata: dict = None):
        docs = self.splitter.create_documents(texts=[raw], metadatas=[metadata])
        if hasattr(self, "db"):
            self.db.add_documents(docs)
        else:
            self.db = FAISS.from_documents(docs, self.model)
            self._init_chain()
        self._save()

    def _index_url(self, url: str, loader_type: Literal["bs", "unstructured"]):
        try:
            if loader_type is None:
                loader_type = self.loader_type
            res = requests.get(url)
            temp_filename = f"{url.split('/')[-1]}.html"
            with open(temp_filename, "w") as f:
                f.write(res.text)
            if self.loader_type == "unstructured":
                loader = UnstructuredHTMLLoader(file_path=temp_filename)
            elif self.loader_type == "bs":
                loader = BSHTMLLoader(file_path=temp_filename)
            else:
                raise ValueError("loader_type must be one of 'bs' or 'unstructured'")
            docs = loader.load()
            os.remove(temp_filename)
            text = docs[0].page_content
            metadata = docs[0].metadata
            docs = self.splitter.create_documents(
                texts=[text], metadatas=[{**metadata, "source": url}]
            )
            if hasattr(self, "db"):
                self.db.add_documents(docs)
            else:
                self.db = FAISS.from_documents(docs, self.model)
                self._init_chain()
            self._save()
            return True
        except Exception as e:
            logging.error(f"Failed to index {url}: {e}")
        return False

    def index_multiple_urls(
        self,
        urls: List[str],
        loader_type: Optional[Literal["bs", "unstructured"]] = None,
    ):
        for url in urls:
            is_successful = self._index(url, loader_type=loader_type)
            if not is_successful:
                return False
        return True

    def query(self, text: str):
        if not hasattr(self, "chain"):
            raise ValueError("You must index some documents before you can query.")
        if self.retrieval_only:
            return self.chain.retriever.get_relevant_documents(text)
        if self.chain_id == "qa":
            return self.chain.run(text)
        if self.chain_id == "qa_source":
            return self.chain({"question": text})
        raise ValueError(f"Unknown chain_id: {self.chain_id}")

    def _save(self):
        try:
            self.db.save_local(self.faiss_path)
            write_indices_list(index_name=self.index_name, faiss_path=self.faiss_path)
            return True
        except Exception as e:
            logging.error(f"Failed to save FAISS index to {self.faiss_path}: {e}")
        return False


class SongTagger(BaseTagger):
    PROMPT = """Here are the song titles along with the artist name(s):\n{songs}"""
    HISTORY = [
        HumanMessage(
            content="I want you to generate tags for the songs that I provide in the next input. The tags must be musically and culturally informative as I plan to use them to find songs and make playlists. Generate around 5 tags per song. The output must be in the format of a JSON-serialized list of lists where each element corresponds to one song. Return only the list. Do you understand?"
        ),
        AIMessage(
            content="Yes, I understand. Please provide the song inputs and I will generate musically and culturally informative tags for each song as a JSON list of lists and return only the list."
        ),
    ]
    DUMMY = {"name": "Love Me Do", "artist": "The Beatles"}

    def __init__(self):
        super().__init__(llm_id="claude", llm_kwargs_id="tagger")

    def _format_tracks(self, tracks: List[Song]):
        formatted_tracks = "\n".join(
            [f'"{track["name"]}" by {track["artist"]}' for track in tracks]
        )
        return formatted_tracks

    def _verify_output(self, output: list, songs: List[Song]):
        try:
            assert len(songs) + 1 == len(
                output
            ), f"Expected {len(songs) + 1} songs (including dummy), got {len(output)}"
            assert (
                list_depth(output) == 2
            ), f"Expected output to be a list of lists, got {output}"
            return output[1:]
        except:
            print("Output:", output)
            raise

    @retry(3, 0.1)
    def __call__(self, songs: List[Song]):
        formatted_songs_input = self._format_tracks([SongTagger.DUMMY, *songs])
        output = self.generate_with_prompt(
            SongTagger.PROMPT,
            {"songs": formatted_songs_input},
            history=SongTagger.HISTORY,
        )
        parsed_output = self._parse_list(output)
        verified_output = self._verify_output(parsed_output, songs)
        formatted_output = [
            self._normalize_tag(tag)
            for song_tags in verified_output
            for tag in song_tags
        ]
        return formatted_output


def get_readers():
    indices = read_indices_list()
    return {index_name: Freader(**kwargs) for index_name, kwargs in indices.items()}


def add_reader(readers, index_name):
    index_path = f"data/{index_name}"
    reader = Freader(index_name, index_path)
    readers[index_name] = reader
