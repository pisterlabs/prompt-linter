from __future__ import annotations

from langchain.text_splitter import CharacterTextSplitter

from typing import List

import re

class AliTextSplitter(CharacterTextSplitter):
    def __init__(self, pdf: bool = False, **kwargs):
        super().__init__(**kwargs)
        self.pdf = pdf

    def split_text(self, text: str) -> List[str]:
        if self.pdf:
            text = re.sub(r"\n{3,}", r"\n", text)
            text = re.sub('\s', " ", text)
            text = re.sub("\n\n", "", text)
        try:
            from modelscope.pipelines import pipeline
        except ImportError:
            raise ImportError(
                "Could not import modelscope python package. "
                "Please install modelscope with `pip install modelscope`. "
            )
        
        p = pipeline(
            task = "document-segmentation",
            model = 'damo/nlp_bert_document-segmentation_english-base',
            device = "cuda"
        )
        result = p(documents = text)
        sent_list = [i for i in result["text"].split("\n\t") if i]
        return sent_list
    