import os
import openai
import subprocess
import sqlite3
import csv
from django.http import FileResponse
import pandas as pd
import json
import plotly.graph_objects as go
import plotly.offline as pyo
from django.http import HttpResponse
import logging

logger = logging.getLogger(__name__)

def your_function(txt_file_path, db_file_path, api_key, user_query):
    with open(txt_file_path, 'r') as file:
        database_info = file.read()
    openai.api_key = api_key
    user_query = user_query
    chat_completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a sql expert in data analysis"},
            {"role": "user", "content":
             f'''Pretend you are a sql data analsyis,

            My database break down is as follows:

            {database_info}
            I want you to give only the sql code as a output so for example:

            Input: How do i select all the customers first names
            Output:
            SELECT FirstName FROM customers;

            input: Update a albums name to "new name" where its id is 101
            Output:
            UPDATE albums
            SET Title = 'new name'
            WHERE Albumid = 101;

            -----


            {user_query}

        '''}
        ])
    query = chat_completion['choices'][0]['message']['content']
    # Connect to the SQLite database and execute the query
    conn = sqlite3.connect(db_file_path)
    cursor = conn.cursor()
    cursor.execute(query)
    rows = cursor.fetchall()
    conn.close()
    # Write the results to a CSV file
    with open('extracted-data.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(rows)
    # Create a FileResponse to send the CSV file to the client
    response = FileResponse(open('extracted-data.csv', 'rb'))
    response['Content-Disposition'] = 'attachment; filename="extracted-data.csv"'
    return response


def python_visualise(chart_type):
    df = pd.read_csv('extracted-data.csv')
    data_string = df.to_string()
    chat_completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a python expert in data analysis"},
            {"role": "user", "content":
             f'''Pretend you are a python data analsyis,

                I want you to give only the python code as a output so for example:

                Example 1:
                Input: How to print something
                Output: print("hello world")

                Example 2:
                Input: How to create a bar chart
                Output:
                import matplotlib.pyplot as plt
                categories = ['A', 'B', 'C']
                values = [10, 15, 7]
                plt.bar(categories, values)
                plt.show()

                Example 3:
                Input: how to craete a scatter plot in pyton
                Output:
                import matplotlib.pyplot as plt
                x_values = [1, 2, 3, 4, 5]
                y_values = [5, 7, 6, 8, 7]
                plt.scatter(x_values, y_values)
                plt.xlabel('X Values')
                plt.ylabel('Y Values')
                plt.title('Scatter Plot')
                plt.show()

                Example 4:
                Input: Create me a bar chart example using plotpy
                Output:
                import plotly.graph_objects as go
                fruits = ['Apples', 'Oranges', 'Bananas', 'Grapes', 'Berries']
                quantities = [10, 15, 7, 10, 5]
                fig = go.Figure([go.Bar(x=fruits, y=quantities)])
                fig.update_layout(title_text='Fruit Quantities', xaxis_title='Fruit', yaxis_title='Quantity')
                fig.show()



                -----

                Visualise this data using plotly as a {chart_type}: {data_string}

    '''}
        ]
    )
    execute = chat_completion['choices'][0]['message']['content']
    execute_temp = execute.split('\n')
    execute = '\n'.join(execute_temp[:-1])
    with open('python-execute.py', 'w') as f:
        f.write('import json\n')
        f.write(execute)
        f.write('\nprint(fig.to_json())')
    result = subprocess.run(
        ["python", 'python-execute.py'], stdout=subprocess.PIPE)
    stdout = result.stdout.decode()
    fig_json = json.loads(stdout)
    if fig_json:
        fig = go.Figure(fig_json)
        plot_html = pyo.plot(fig, output_type='div')
        return HttpResponse(plot_html)
    else:
        # If there was no stdout, return the stderr to help diagnose the issue.
        return HttpResponse(f"Error executing script: {result.stderr}")

def explain_data(user_query):
    df = pd.read_csv('extracted-data.csv')
    data_string = df.to_string()
    chat_completion = openai.ChatCompletion.create(
    model="gpt-3.5-turbo", 
    messages=[
        {"role": "system", "content": "You are a expert in explaining data analysis to non technical audience"},
        {"role": "user", "content": 
        f''' Given this user query:

            {user_query}

            And this output:
            {data_string}

            Explain the output data to me, dont go into much technical details, your tagret audience is non technical. Just explain the output data and context

    '''}
            ]
        )

    query_explain = chat_completion['choices'][0]['message']['content']
    logger.debug(f'Generated explaination: {query_explain}')

    return query_explain