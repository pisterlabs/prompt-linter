from flask import Flask, jsonify, request, render_template

from data_processing import SlackDataProcessor, NotionDataProcessor
from langchain_model import LangChain


class LangchainAPI:
    def __init__(self, chain: LangChain):
        self.app = Flask(__name__)
        self.chain = chain
        self.chat_history = []

        @self.app.route('/text')
        def text():
            return self.chain.data

        @self.app.route('/chat')
        def chat():
            query = request.args.get('message')
            response = self.chain.chat(query)
            return jsonify(response)

        @self.app.route('/')
        def index():
            with open("data/test_queries.txt") as f:
                questions = f.read().split("\n")
            # escape quotes
            questions = [q.replace('"', '\\"').replace("'", "\\'") for q in questions]
            return render_template('index.html', questions=questions, full_text=self.chain.data)

    def run(self):
        self.app.run(debug=True)


if __name__ == '__main__':
    slack_data_processor = SlackDataProcessor("data/slack_data.txt")
    slack_data = slack_data_processor.process()

    notion_data_processor = NotionDataProcessor("data/notion_data.txt")
    notion_data = notion_data_processor.process()

    texts = slack_data + notion_data
    lang_chain = LangChain(text=texts, chain_type="stuff", embeddings_type="openai")

    api = LangchainAPI(lang_chain)
    api.run()
