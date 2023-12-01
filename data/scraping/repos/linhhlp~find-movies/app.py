"""This serves as Web Endpoints for Find-movies service."""

import uuid

import cohere
import flask

from cred import (
    ASTRA_CLIENT_ID,
    ASTRA_CLIENT_SECRET,
    SECURE_CONNECT_BUNDLE_PATH,
    API_key,
)
from logtool import Logger
from utls import Cassandra, convert_text_2_vect, shorten

# from flask import session


# get free Trial API Key at https://cohere.ai/
co = cohere.Client(API_key)

KEYSPACE_NAME = "demo"
TABLE_NAME = "movies_35K_vectorized"

db_info = {
    "secure_bundle_path": SECURE_CONNECT_BUNDLE_PATH,
    "client_id": ASTRA_CLIENT_ID,
    "client_secret": ASTRA_CLIENT_SECRET,
    "keyspace": KEYSPACE_NAME,
    "table_name": TABLE_NAME,
}

config = {
    "protocol_version": 4,
}

cass = Cassandra(db_info, config)

logger = Logger("movie_logger", cass)

# Start web framework
app = flask.Flask(__name__)
app.secret_key = "slkghsjklghslkgh383i232@$@$95ji32($5)"  # Random string


@app.route("/", methods=["GET", "POST"])
def find_movies():
    """Find movies."""
    if "uid" not in flask.session:
        flask.session["uid"] = uuid.uuid4()

    data = {}
    candidate = content = ""
    if flask.request.method == "POST" and flask.request.values.get("content"):
        content = flask.request.values.get("content")
        method = flask.request.values.get("method")
        if method != "plot_vector_1024":
            method = "plot_summary_vector_1024"
        limit = 5
        # Convert Text to Vector
        vec = convert_text_2_vect(co, [content], "embed-english-light-v2.0")[0]

        # Prompt Engineering: Building prompt
        prompt = (
            f"Based on the plots of {limit} movie"
            + ("s" if limit > 1 else "")
            + f' below, suggest one movie which "{content}": \n'
        )

        # Vector Search: Finding nearest neighbors from Cassandra DB
        for i, row in enumerate(
            cass.execute(
                f"""
SELECT year, title, director, cast, genre, wiki_link, plot, plot_summary 
 FROM {KEYSPACE_NAME}.{TABLE_NAME} ORDER BY {method} ANN OF %s LIMIT {limit}
""",
                [vec],
            ),
            1,
        ):
            data[i] = row
            prompt += f'Movie {i}: title is : "{row.title}" in ({row.year})'
            if row.director != "Unknown":
                prompt += f", by director {row.director}"
            prompt += f", genre is {row.genre}"
            prompt += ", the plot is: \n" + shorten(row.plot, 500) + "\n"

        # Prompt Engineering: Generate top results from a list of movies
        generated_text = co.generate(
            # model="command",  #
            model="command-nightly",
            prompt=prompt,
            max_tokens=60,
            temperature=0.5,
            # truncate="END",  # maximum token length will discard the end
        )

        candidate = generated_text.generations[0].text
        logger.upload(candidate + ";====;" + logger.data_2_str(data), flask)

    return flask.render_template(
        "index.html", content=content, candidate=candidate, data=data
    )


# if __name__ == "__main__":
#    app.run(host="0.0.0.0")
