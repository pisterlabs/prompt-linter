
from flask import Flask, render_template, request, jsonify


from Searcher import query_processing, stopword_check, bool_AND, calculate_Score, print_search


import time


import json


import openai


import requests


from bs4 import BeautifulSoup

# Set the OpenAI api key
openai.api_key = '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'


app = Flask(__name__)

# Initialize the query index
query_index = {}

@app.route('/', methods=['GET', 'POST']) #defines a route for the root URL
def search():
    # Check if the request method is POST
    if request.method == 'POST':
        # Get the query from the form
        query = request.form['query']

        # Start the timer for query processing
        start_time = time.time()

        # Process the query, removing stopwords and calculating scores
        query_index = query_processing(query)
        query_index = stopword_check(query_index)
        common_subkeys_list = bool_AND(query_index)
        scores = calculate_Score(query_index, common_subkeys_list)

        # Get the search results
        results = get_search_results(scores)

        # Stop the timer and calculate the elapsed time
        end_time = time.time()
        elapsed_time = round((end_time - start_time) * 1000, 2)

        # Return the query, results, and time in JSON format
        return jsonify({'query': query, 'results': results, 'time': elapsed_time})

    # If the request method is GET, render the search form
    return render_template('index.html')


@app.route('/reset', methods=['POST']) #This should work only when user posts the reset button
def reset():
    # Reset the query index and result list
    global query_index, result_list
    query_index.clear()
    result_list = []
    common_subkeys_list = []
    print("Query index reset successful.")
    # Return a success message
    return jsonify({'message': 'Query index reset successful.'})


def get_search_results(scores):
    result_list = []
    counter = 5
    urldict = {}

    # Load the urls from the json file
    with open("docIDs.json", "r") as file:
        urldict = json.load(file)

    # Invert the url dictionary
    urldict2 = {value: key for key, value in urldict.items()}
    # Sort the scores in descending order
    sorted_scores = {k: v for k, v in sorted(
        scores.items(), key=lambda item: item[1], reverse=True)}

    # Get the top 5 search results
    for key in sorted_scores:
        if counter <= 0:
            break
        result_list.append(urldict2[key])
        counter -= 1

    # Return the result list
    return result_list




@app.route('/summary', methods=['POST'])  #Works only when user select the Summary button
def summary():
    # Get the url from the form
    url = request.form['url']
    # Generate a prompt for the GPT-3 model
    prompt = f"Summarize the contents of the web page at {url}."
    # Use the GPT-3 model to generate a summary
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": "What is the summary?"},
        ],
    )
    # Get the summary from the response
    summary = response.choices[0].message.content
    # Return the summary in JSON format
    return jsonify({'summary': summary})



if __name__ == '__main__':
    app.run(debug=True)
