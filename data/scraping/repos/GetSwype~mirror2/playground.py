from model.core import Conversation, Message
from model.helpers import timestamp_to_datetime
from model.messageparser import MessageParser
from llama_index import playground, GPTTreeIndex, GPTListIndex, GPTSimpleVectorIndex, Document, LLMPredictor
from langchain import OpenAI
import datetime

from llama_index import GPTSimpleVectorIndex


if __name__ == "__main__":
    cnv = Conversation()
    messages = MessageParser("imessage", "chat.txt").parse()
    chunks = [messages[i:i + 10] for i in range(0, len(messages), 10)]
    documents = []
    llm_predictor = LLMPredictor(llm=OpenAI(temperature=0.7, model_name="text-curie-001"))
    for chunk in chunks:
        documents.append(Document("\n".join(str(message) for message in chunk)))
    plg = playground.Playground(
        [
            GPTTreeIndex(documents, llm_predictor=llm_predictor),
            GPTListIndex(documents, llm_predictor=llm_predictor),
            GPTSimpleVectorIndex(documents, llm_predictor=llm_predictor)
        ]
    )
    query = Message("srikanth", "Do you remember what I said about corsair?", datetime.datetime.now().timestamp())
    prompt = open("model/prompts/memory.txt", "r").read()
    prompt.replace(
        "<<TIMESTAMP>>", timestamp_to_datetime(query.timestamp)
    )
    prompt.replace(
        "<<AUTHOR>>", query.author
    )
    prompt.replace(
        "<<CONTENT>>", query.text
    )
    print(plg.compare(prompt))