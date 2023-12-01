from langchain.utilities import SQLDatabase

db = SQLDatabase.from_uri("sqlite:///nba_roster.db", sample_rows_in_table_info=0)


def get_schema(_):
    return db.get_table_info()


def run_query(query):
    return db.run(query)

print(get_schema)