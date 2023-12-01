from sqlalchemy import create_engine
from sqlalchemy import text
import pandas as pd

def dataframe_to_database(df, table_name):
    """Convert a pandas dataframe to a database.
        Args:
            df (dataframe): pd.DataFrame which is to be converted to a database
            table_name (string): Name of the table within the database
        Returns:
            engine: SQLAlchemy engine object
    """
    engine = create_engine(f'sqlite:///:memory:', echo=False)
    df.to_sql(name=table_name, con=engine, index=False)
    return engine
 
def handle_response(response):
    """Handles the response from OpenAI.

    Args:
        response (openAi response): Response json from OpenAI

    Returns:
        string: Proposed SQL query
    """
    query = response["choices"][0]["text"]
    if query.startswith(" "):
        query = "Select"+ query
    return query

def execute_query(engine, query, df):
    """Execute a query on a database.

    Args:
        engine (SQLAlchemy engine object): database engine
        query (string): SQL query

    Returns:
        list: List of tuples containing the result of the query
    """
    with engine.connect() as conn:
        result = conn.execute(text(query))
        rows = result.fetchall()
        print("df.columns========st======>")
        
       
        index_list = df.columns.tolist()
        
        converted_tuple = tuple(index_list)
        print(converted_tuple)
        print("df.columns=========en=====>")
        # converted_data = [dict(zip(",".join(str(col) for col in index_list), row)) for row in rows]
        converted_data = [dict(zip(converted_tuple, row)) for row in rows]
        return converted_data
