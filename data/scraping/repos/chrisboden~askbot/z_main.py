# Import programs that are installed via pip
import openai
from flask import Flask, request, jsonify, render_template, make_response

# Import stuff that shouldn't need to be installed via pip
import os
import time
import json
import urllib.parse
from datetime import datetime
from pathlib import Path

# Import from other python files in our code
import sqlite3
from setup_index import index, metadata
from get_answers import searchq
from get_embedding import get_embedding
from get_completion import completions
from add_history import add_history

app = Flask(__name__)

# Set up the various file path variables so that the paths are relative no matter the environment
cwd = Path(__file__).parent.resolve()
data_path = cwd.joinpath('data', 'main')
sources_json_path = data_path.joinpath('sources.json')
prompt_txt_path = cwd.joinpath('prompt_template.txt')

@app.route("/")
def home():
  return render_template("home.html")
  print("template rendered")

@app.route("/search", methods=["POST"])
def search():
  # Get current time
  current_time = datetime.now()
  current_time = current_time.strftime("%Y-%m-%dT%H:%M:%S")
  # Start the timer
  start_time = time.perf_counter()
  # Take the query input from the form
  query = request.form["query"]
  # Generate embedding for user query
  query_embedding = get_embedding(query)
  # Call the search function with the required arguments
  results_list = searchq(index, query_embedding, 'metadata.db')
  #print(f"The query results are {results_list}")

  # Call completions function and pass in the query and results_list
  if not results_list:
      return "Aborting completion because no database results"
  completion, prompt_tokens_count, context_length, prompt, max_tokens, results_list, gpt_temperature, gpt_model = completions(query, results_list)

  # Stop the timer
  end_time = time.perf_counter()
  # Calculate elapsed time
  elapsed_time = end_time - start_time

  # Uncomment here if you want to see results printed in console
  # Print the various outputs to test
  # print(f"The starttime timestamp is {start_time}")
  # print(f"The query is {query}")
  # print(f"The query embedding is{query_embedding}")
  # print(f"The prompt is {prompt}")
  # print(f"The context length is {context_length}")
  # print(f"The total prompt tokens is {prompt_tokens_count}")
  # print(f"The prompt is {completion}")
  print(f"The max tokens remaining is {max_tokens}")

  #Add a record to the ask_history json file
  add_history(current_time, query, results_list, prompt, completion, elapsed_time, gpt_temperature, gpt_model)

  # print(f"Elapsed time: {elapsed_time:.4f} seconds")

  # Return response as JSON
  return jsonify(list(completion))

@app.route("/mentors", methods=["GET"])
def get_authors():
  # Set the path to the json file
  json_file = sources_json_path

  # Read the data from the json file
  with open(json_file, 'r') as f:
    data = json.load(f)

  # Initialize an empty string to store the results
  results = ''

  # Iterate through the records in the data
  for record in data:
    # Extract the data from the record
    book_author = record['bookAuthor']
    book_title = record['bookTitle']
    # Build the result string
    result = f'<p><b>{book_author}</b>: {book_title}</p>'
    # Append the result to the results string
    results += result
    
  return results

# Route for getting the prompt text
# Note you can change the prompt text in the ui. If you change on the file system, you'll need to relaunch the flask app.
@app.route("/get-prompt")
def get_prompt():
  try:
    # Read the prompt text from the prompt.txt file. 
    with open(prompt_txt_path, "r") as f:
      prompt_text = f.read()
    # Return the prompt text
    return prompt_text
  except Exception as e:
    return f"Error getting prompt text: {e}"


# Route for saving the edited prompt text
@app.route("/save-prompt", methods=["POST"])
def save_prompt():
  try:
    # Get the edited prompt text from the form data
    edited_prompt = request.form['edited_prompt']

    # Write the edited prompt text to the prompt.txt file
    with open(prompt_txt_path, "w") as f:
      f.write(edited_prompt)
    # Return success message
    return "Prompt text updated successfully"
  except Exception as e:
    # Return error message
    return f"Error updating prompt text: {e}"


if __name__ == "__main__":
  app.run()

# This is for livereloading the website when you're in dev mode
# from livereload import Server
# server = Server(app.wsgi_app)
# server.watch('static/custom.css')
# server.watch('templates/home.html')
# server.serve(open_url_delay=True)