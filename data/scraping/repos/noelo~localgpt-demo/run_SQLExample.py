import os
import openai
from langchain import OpenAI, SQLDatabase, SQLDatabaseChain
from chainlit import AskUserMessage, Message, on_chat_start, on_message, langchain_factory

@on_chat_start
async def main():
        await Message(
            content=f"Ask questions of the database",
        ).send()


@langchain_factory(use_async=False)
def load_model():
    openai.api_key = os.environ["OPENAI_API_KEY"]
    db = SQLDatabase.from_uri("postgresql+psycopg2://postgres:@127.0.0.1/dellstore2")
    llm = OpenAI(temperature=0, verbose=True)
    db_chain = SQLDatabaseChain.from_llm(llm, db, verbose=True,top_k=50)
    return db_chain
