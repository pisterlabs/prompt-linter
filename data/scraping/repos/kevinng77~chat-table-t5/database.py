import pandas as pd
from langchain import SQLDatabase
from sqlalchemy import (
    text,
)
import sqlite3


class PdSQLDatabase(SQLDatabase):
    def sql2df(self, fetched_result):
        if len(fetched_result) == 1:
            return fetched_result[0][0]
        return pd.DataFrame(fetched_result)

    def run(self, command: str, fetch: str = "all") -> str:
        """Execute a SQL command and return a string representing the results.

        If the statement returns rows, a string of the results is returned.
        If the statement returns no rows, an empty string is returned.
        """
        with self._engine.begin() as connection:
            if self._schema is not None:
                connection.exec_driver_sql(f"SET search_path TO {self._schema}")
            cursor = connection.execute(text(command))
            if cursor.returns_rows:
                if fetch == "all":
                    result = cursor.fetchall()
                    return self.sql2df(result)
                elif fetch == "one":
                    result = cursor.fetchone()[0]  # type: ignore
                    return result

                else:
                    raise ValueError("Fetch parameter must be either 'one' or 'all'")
        return ""


def init_sql_db(table_name="my_table", df=None, database_path="your_data.db"):
    uri = "sqlite:///" + database_path
    db = PdSQLDatabase.from_uri(uri)
    if table_name not in db._all_tables:
        # if df (a pandas DataFrame) provided, Store it in a sqlite.db
        conn = sqlite3.connect(database_path)
        df.to_sql(table_name, conn, if_exists="replace", index=False)
        conn.commit()
        conn.close()
        db = PdSQLDatabase.from_uri(uri)
    return db
