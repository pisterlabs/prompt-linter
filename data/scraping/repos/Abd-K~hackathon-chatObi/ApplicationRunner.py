from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin

from ChatObi import ChatObi
from LlamaIndex import construct_index
app = Flask(__name__)

CHAT_HISTORY_PATH = "chat_history.json"
chatObi = ChatObi()

def setupBot():
    CORS(app, origins='*')
    chatObi.load_chat_history(CHAT_HISTORY_PATH)
    chatObi.clear_chat_history()
    # construct_index()

setupBot()

@cross_origin(origins='*')
@app.route('/chatObi')
def chat():
    input_query = request.args.get('query')
    response = chatObi.query_index(input_query)
    chatObi.save_chat_history(CHAT_HISTORY_PATH)
    print(f"Bot: {response}")
    return jsonify(response)

if __name__ == '__main__':
    app.run(port=5000, debug=True)
