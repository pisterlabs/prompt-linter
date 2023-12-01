import pandas as pd
import sqlite3
import os
import yaml
from langchain import SQLDatabase
config = yaml.safe_load(open("config.yaml"))


def get_table_html(tablename):
    conn = sqlite3.connect(config["DATABASE_NAME"])
    dataframe = pd.read_sql_query(
        "SELECT * FROM {} LIMIT 10".format(tablename), conn)
    conn.close()
    table_html_header = '\n'.join(
        [f'<th scope="col">{ex}</th>' for ex in dataframe.columns.tolist()]
    )
    table_html_body = '\n'.join(
        ['<tr>{rows}</tr>'.format(
            rows='\n'.join([f'<td>{col}</td>' for col in row])
        ) for row in dataframe.values
        ])

    return f'''<div class = "example-table-wrapper">
    <table class="table table-bordered table-striped table-hover">
    <thead class = "thead-dark">
    <tr>
    {table_html_header}
    </tr>
    </thead>
    <tbody>
    {table_html_body}
    </tbody>
    </table>
    </div>'''

def create_db(tablename, dataframe):
    conn = sqlite3.connect(config["DATABASE_NAME"])
    dataframe.to_sql(tablename, conn, if_exists = "replace", index = False)
    conn.close()
    return SQLDatabase.from_uri(
        database_uri=f'sqlite:///{config["DATABASE_NAME"]}'
    )

def sanitize_tablename(filename):
    x = os.path.split(filename)[-1]
    x = x.lower()
    x = re.sub(r'^\d+', '', x)
    x = re.sub(r'\..*', '', x)
    return x.replace(' ', '_').replace('-', '_')
