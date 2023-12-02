import os
from sqlalchemy import create_engine, MetaData, text
from llama_index import LLMPredictor, ServiceContext, SQLDatabase, VectorStoreIndex
from llama_index.indices.struct_store import SQLTableRetrieverQueryEngine
from llama_index.objects import SQLTableNodeMapping, ObjectIndex, SQLTableSchema
from langchain import OpenAI

# OpenAI API key:
# os.environ["OPEN_AI_KEY"] = "your_api_key"

# set up database:
url = "sqlite:///atradebot.db"
db = create_engine(url)

# connection:
conn = db.connect()

# Print first 5 rows of stocks table to check connection:
# # SQL query:
# sql_query = text("SELECT * FROM stocks LIMIT 5;")
#
# try:
#     results = conn.execute(sql_query)
#     rows = results.fetchall()  # Fetch all rows from the result
# except Exception as e:
#     print(f"Error executing query: {e}")
#     rows = None
#
# if rows:
#     for row in rows:
#         print(row)
# else:
#     print("No results returned from query")
#
# conn.close()

# loading table definitions:
metadata_obj = MetaData()
metadata_obj.reflect(db)

# print(metadata_obj.tables.keys())

# create a database object:
db_obj = SQLDatabase(db)

# table node mapping:
table_node_mapping = SQLTableNodeMapping(db_obj)

table_schema_objs = []
for table_name in metadata_obj.tables.keys():
    table_schema_objs.append(SQLTableSchema(table_name=table_name))

obj_index = ObjectIndex.from_objects(
    table_schema_objs,
    table_node_mapping,
    VectorStoreIndex,
)


# llm:
llm = OpenAI(model_name="gpt-4", max_tokens=6000)

# llm predictor:
llm_predictor = LLMPredictor(llm=llm)

# service context:
service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor)

# query engine:
query_engine = SQLTableRetrieverQueryEngine(
    db_obj,
    obj_index.as_retriever(similarity_top_k=1),
    service_context = service_context,
)




