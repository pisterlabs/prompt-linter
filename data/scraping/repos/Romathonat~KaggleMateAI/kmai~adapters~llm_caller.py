from langchain.embeddings import OpenAIEmbeddings

from kmai.ports.illm_caller import ILLMCaller


class LLMCaller(ILLMCaller):
    def __init__(self) -> None:
        super().__init__()
        self.embeddings_model = OpenAIEmbeddings()

    def get_embeddings(self, text_list: list[str]) -> list[list]:
        return self.embeddings_model.embed_documents(text_list)
