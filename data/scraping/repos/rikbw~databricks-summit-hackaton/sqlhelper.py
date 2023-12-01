from langchain import SQLDatabase, SQLDatabaseChain
from langchain.chat_models import ChatOpenAI

db = SQLDatabase.from_uri("sqlite:///Chinook_Sqlite.sqlite")
llm = ChatOpenAI(temperature=0)
db_chain = SQLDatabaseChain.from_llm(llm, db, verbose=True)

def user_typed_prompt(prompt):
    print(db_chain.run(prompt))
