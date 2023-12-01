
from typing import List

import re
from langchain.document_loaders import UnstructuredFileLoader
from langchain.text_splitter import CharacterTextSplitter


class ChineseTextSplitter(CharacterTextSplitter):
    def __init__(self, pdf: bool = False, **kwargs):
        super().__init__(**kwargs)
        self.pdf = pdf

    def split_text(self, text: str) -> List[str]:
        if self.pdf:
            text = re.sub(r"\n{3,}", "\n", text)
            text = re.sub('\s', ' ', text)
            text = text.replace("\n\n", "")
        sent_sep_pattern = re.compile(
            '([﹒﹔﹖﹗．。！？]["’”」』]{0,2}|(?=["‘“「『]{1,2}|$))') 
        sent_list = []
        for ele in sent_sep_pattern.split(text):
            if sent_sep_pattern.match(ele) and sent_list:
                sent_list[-1] += ele
            elif ele:
                sent_list.append(ele)
        return sent_list



def load_file(filepath):
    print("filepath:",filepath)
    if filepath.endswith(".md"):
        loader = UnstructuredFileLoader(filepath, mode="elements")
        docs = loader.load()
    elif filepath.endswith(".pdf"):
        loader = UnstructuredFileLoader(filepath)
        textsplitter = ChineseTextSplitter(pdf=True)
        docs = loader.load_and_split(textsplitter)
    else:
        loader = UnstructuredFileLoader(filepath, mode="elements")
        textsplitter = ChineseTextSplitter(pdf=False)
        docs = loader.load_and_split(text_splitter=textsplitter)
    return loader,docs