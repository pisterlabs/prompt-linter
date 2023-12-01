# Create a web interface that allows the user to enter a search term in the url and returns the most similar documents
# use Flask 

from flask import Flask, request, render_template
from modules.utils import carregar_credenciais
from modules.utils import agente, agente_index
from langchain.vectorstores import FAISS


app = Flask(__name__)

@app.route('/')
def index():
    results = agente_index()
    # return the results
    return render_template('results.html', results=results)


@app.route('/search', methods=['GET'])
def search():
    # get the search term from the url
    query = request.args.get('q')
    results = agente(query)
    # return the results
    return render_template('results.html', query=query, results=results)

if __name__ == '__main__':
    app.run(debug=True)
