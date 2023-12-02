import json
import xml.etree.ElementTree as ET

from pathlib import Path
from typing import List, Union
from dataclasses import dataclass
from langchain.prompts import PromptTemplate

from GPTagger.validators import *
from GPTagger.indexer import Indexer, Tag
from GPTagger.textractor import Textractor
from GPTagger.logger import log2cons, log2file, setup_log2file


@dataclass
class NerConfig:
    tag_name: str
    # tagger cfgs
    nr_calls: int = 1
    use_tool: bool = True
    model: str = "gpt-3.5-turbo-0613"
    # indexer cfgs
    token_threshold: int = 80
    phrase_threshold: int = 85
    # validator cfgs
    tag_regex: str = None
    tag_max_len: int = 128
    # paths
    log_dir: Path = None
    export_dir: Path = None


class NerPipeline:
    """
    The pipeline is used for NER annoation on text
    """

    def __init__(
        self,
        tag_name: str,
        nr_calls: int = 1,
        use_tool: bool = True,
        model: str = "gpt-3.5-turbo",
        token_threshold: int = 80,
        phrase_threshold: int = 85,
        tag_regex: str = None,
        tag_max_len: int = None,
        log_dir: Union[Path, str] = None,
        export_dir: Union[Path, str] = None,
    ) -> None:
        log2cons.info("NER pipeline for <%s> recognition", tag_name)
        self.tag_name = tag_name
        self.export_dir = export_dir

        self.textractor = Textractor(
            model=model,
            use_tool=use_tool,
            num_of_calls=nr_calls,
        )

        self.indexer = Indexer(token_threshold, phrase_threshold)

        self.validators = []
        if tag_max_len:
            self.validators.append(LengthValidator(tag_max_len))
        if tag_regex:
            self.validators.append(RegexValidator(tag_regex))

        if log_dir:
            log_dir = Path(log_dir) if isinstance(log_dir, str) else log_dir
            log_dir.mkdir(exist_ok=True)
            setup_log2file(log_dir / f"filter.log")

    @classmethod
    def from_config(cls, config: NerConfig) -> "NerPipeline":
        return cls(**config.__dict__)

    def add_validator(self, validator: BaseValidator):
        self.validators.append(validator)

    def __call__(self, text: str, template: PromptTemplate, fname: str = None) -> List[Tag]:
        # Step 1. Extraction
        extractions = self.textractor(text, template)
        tags = self.indexer.index(extractions, text, fname)
        log2cons.info("Extract %d <%s> tags.", len(tags), self.tag_name)
        # Step 2. Validation
        tags = self._validate(tags, fname)
        log2cons.info("Validate %d <%s> tags.", len(tags), self.tag_name)
        # Step 3. Export
        self._export(text, tags, fname)

        return tags

    def _validate(self, tags: List[Tag], fname: str = None) -> List[Tag]:
        if not tags:
            return []

        tags_filtered = []

        for tag in tags:
            flag = True
            for validator in self.validators:
                if not validator(tag.text):
                    # extraction is not validated
                    flag = False
                    log = {
                        "filter_name": type(validator).__name__,
                        "text": tag.text,
                        "fname": fname,
                    }
                    log2file.info(f"{json.dumps(log, ensure_ascii=False)}")
                    break
            if flag:
                tags_filtered.append(tag)

        if len(tags_filtered) <= 1:
            return tags_filtered

        # Remove overlapping extractions after validation
        tags_wo_overlap = self.indexer.resolve_overlap(tags_filtered, fname)

        return tags_wo_overlap

    # TODO add test for exporting
    def _export(self, text: str, tags: List[Tag], fname: str):
        if not self.export_dir:
            return

        path = self.export_dir / fname

        root = ET.Element("begin")
        root.set("id", fname)

        if not tags:
            root.text = text

        for i, tag in enumerate(tags):
            if i == len(tags) - 1:
                next = len(text)
            else:
                next = tags[i + 1].start
            if not root.text:
                root.text = text[: tag.start]
            element = ET.SubElement(root, self.tag_name, id=str(i))
            element.text = tag.text
            element.tail = text[tag.end : next]

        tree = ET.ElementTree(root)
        tree.write(path, encoding="utf-8")
