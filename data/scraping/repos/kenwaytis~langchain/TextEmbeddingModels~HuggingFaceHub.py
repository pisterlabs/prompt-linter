from langchain.embeddings import HuggingFaceEmbeddings


class TextEmbeddings():
    def __init__(self) -> None:
        self.model_name = "GanymedeNil/text2vec-large-chinese"
        self.model_kwargs = {"device":"cpu"}
        self.encode_kwargs = {"normalize_embeddings":False}

    def initmodel(self):
        hf = HuggingFaceEmbeddings(
            model_name=self.model_name,
            model_kwargs=self.model_kwargs,
            encode_kwargs=self.encode_kwargs
        )
        return hf
    
if __name__ == "__main__":
    hf = TextEmbeddings().initmodel()
    text = '你好'
    querty_result = hf.embed_query(text)
    doc_result = hf.embed_documents([text])

    print(querty_result)
    print("***************")
    print(doc_result)
