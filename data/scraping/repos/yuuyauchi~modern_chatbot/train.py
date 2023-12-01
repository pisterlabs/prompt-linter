# from models.langchain_model import LangChainChatBot

from models.llamaindex_model import LlamaindexChatBot


def train():
    model = LlamaindexChatBot()
    model.read_data()
    model.preprocess()
    model.generate_engine()
    # model.evaluate("llamaindex_result.csv")


if __name__ == "__main__":
    train()
