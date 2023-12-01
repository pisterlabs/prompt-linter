from flask import Flask, request, send_file, jsonify
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import openai
import tempfile
import os
from flask_cors import CORS

app = Flask(__name__)
CORS(app)
data = pd.read_csv("data.csv")
def plotter(data, plot_type, time_start, time_end, column_name):
    req_data = data[(data['Year'] >= time_start) & (data['Year'] <= time_end)]
    plt.figure(figsize=(8, 6))
    if "point" in plot_type.lower():
        sns.pointplot(x=req_data["Year"], y=req_data[column_name])
    elif "bar" in plot_type.lower():
        sns.barplot(x=req_data["Year"], y=req_data[column_name])
    elif "pie" in plot_type.lower():
        colors = sns.color_palette('pastel')[0:5]
        plt.pie(req_data[column_name], labels=req_data["Year"], colors=colors)
    plt.xlabel('Year')
    plt.ylabel(column_name)
    plt.title(f'{plot_type.capitalize()} of {column_name} ({time_start}-{time_end})')
    plt.xticks(rotation=90)
    plt.tight_layout()
    temp_dir = tempfile.gettempdir()
    temp_file = os.path.join(temp_dir, 'temp_figure.png')
    plt.savefig(temp_file)
    plt.close()

    return temp_file

api_key = "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"
openai.api_key = api_key
def extract_categories(prompt_text):
    prompt = "Given the following statement, identify the categories for column_name, time_start, time_end, and plot_type:\n\n"\
             "\"" + prompt_text + '"'
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=prompt,
        max_tokens=50,
        temperature=0.6,
        stop=None
    )
    categories = response.choices[0].text.strip().split('\n')
    column_name = categories[0][13:]
    column_name = column_name.replace(" ", "")
    time_start = int(categories[1][12:])
    time_end = int(categories[2][10:])
    plot_type = categories[3][11:]
    plot_type = plot_type.lower()
    plot_type = plot_type.replace(" ", "")
    if 'plot' not in plot_type:
        plot_type = plot_type + 'plot'
    return column_name, time_start, time_end, plot_type

@app.route('/generate_plot', methods=['POST'])
def generate_plot():
    try:
        request_data = request.form
        prompt_text = request_data.get('prompt_text')
        column_name, time_start, time_end, plot_type = extract_categories(prompt_text)
        temp_file_path = plotter(data, plot_type, time_start, time_end, column_name)
        return send_file(temp_file_path, mimetype='image/png')
    except Exception as e:
        return jsonify({'error': str(e)})
if __name__ == '__main__':
    app.run(debug=True)
