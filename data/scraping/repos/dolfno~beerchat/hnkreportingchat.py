import json
from langchain import OpenAI, SQLDatabase, SQLDatabaseChain
from dbprompt import prompt_template, TABLE_INFO
from dotenv import load_dotenv

load_dotenv()

from flask import Flask, jsonify, request
from flask_cors import CORS

app = Flask(__name__)
CORS(app)


class HNKReportingChat:
    """HNKReportingChat class can respond to chat queries and returns the given answer using the data in a postgress db"""

    rprt_uri = "postgresql://root:test@localhost:5432/reporting"

    def __init__(self):
        self._db_chain = None

    def get_answer(self, query: str) -> str:
        return self.db_chain.run(query)

    @property
    def db_chain(self):
        if self._db_chain:
            return self._db_chain
        db = SQLDatabase.from_uri(
            self.rprt_uri,
            schema="reporting",
            include_tables=["datapoints"],
            custom_table_info={"datapoints": TABLE_INFO},
        )
        llm = OpenAI(temperature=0, verbose=True)
        self._db_chain = SQLDatabaseChain.from_llm(llm, db, prompt=prompt_template, verbose=True, top_k=3)
        return self._db_chain


chat = HNKReportingChat()


@app.route("/", methods=["GET"])
def index():
    return {}


@app.route("/metrics", methods=["POST"])
def metrics():
    return jsonify(
        [
            {
                "value": "askquestion",
                "payloads": [
                    {
                        "name": "q",
                        "type": "input",
                        "placeholder": "Put question, eg: How much beer was produced in Belgium yesterday?",
                        "reloadMetric": True,
                        "width": 100,
                    }
                ],
            }
        ]
    )


@app.route("/metric-payload-options", methods=["POST"])
def metric_payload_options():
    return jsonify([{"label": "question", "value": "question"}])


# Route for handling GET requests to execute the method
@app.route("/query", methods=["POST"])
def execute():
    query = json.loads(request.data.decode("UTF-8")).get("targets")[0].get("payload").get("q")
    answer = chat.get_answer(query)

    return jsonify(
        [
            {
                "target": "answer",
                "datapoints": [
                    [answer],
                ],
            }
        ]
    )


if __name__ == "__main__":
    app.run(debug=True)
