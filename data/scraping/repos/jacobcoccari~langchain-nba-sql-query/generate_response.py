# import load_dotenv()
from dotenv import load_dotenv
load_dotenv()

# Creating the connection to the model.
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough

# def get_schema(db):
#     print(type(db.get_table_info()))
#     return db.get_table_info()

def generate_assistant_response(user_question, db):
    llm = ChatOpenAI()

    template = """Based on the table schema below, write a SQL query that would answer the user's question:
    {schema}

    Question: {question}
    SQL Query:"""
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "Given an input question, convert it to a SQL query. No pre-amble."),
            ("human", template),
        ]
    )

    def get_schema(_):
        return db.get_table_info()

    sql_response = (
        RunnablePassthrough.assign(schema=get_schema)
        | prompt
        | llm.bind(stop=["\nSQLResult:"])
        | StrOutputParser()
    )
    template = """Based on the table schema below, question, sql query, and sql response, write a natural language response:
    {schema}

    Question: {question}
    SQL Query: {query}
    SQL Response: {response}"""
    prompt_response = ChatPromptTemplate.from_template(template)

    full_chain = (
        RunnablePassthrough.assign(query=sql_response)
        | RunnablePassthrough.assign(
            schema=get_schema,
            response=lambda x: db.run(x["query"]),
        )
        | prompt_response
        | llm
    )
    response = full_chain.invoke({"question": user_question})
    return response.content
