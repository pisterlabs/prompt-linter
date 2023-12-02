import os
import tempfile
from openai.error import AuthenticationError

from flask import Flask, request, jsonify, make_response

from .librarian import Librarian
from .book_keeper import BookKeeper

app = Flask(__name__, static_folder="../web/dist/")


@app.route("/")
def index():
    return app.send_static_file("index.html")


@app.route("/api/books", methods=["GET"])
def list_books():
    return jsonify(BookKeeper.instance().list_books())


@app.route("/api/books", methods=["POST"])
def upload_book():
    name = request.form.get("name")
    if not name:
        raise ValueError("title is required")

    if "book" not in request.files:
        raise ValueError("invalid book file")

    if request.files["book"].filename == "":
        raise ValueError("book is empty")

    ext_name = os.path.splitext(request.files["book"].filename)[1]
    with tempfile.NamedTemporaryFile(suffix=ext_name) as f:
        print(f.name)
        request.files["book"].save(f.name)
        book_id = BookKeeper.instance().add_book(name, f.name)

    resp = {
        "book_id": book_id,
        "name": name,
    }
    return jsonify(resp)


@app.route("/api/books/<book_id>", methods=["DELETE"])
def delete_book(book_id):
    BookKeeper.instance().delete_book(book_id)
    return jsonify({"status": "success"})


@app.route("/api/books/<book_id>/ask", methods=["POST"])
def ask(book_id):
    question = request.args.get("q")
    if not question:
        raise ValueError("q is required")

    librarian = BookKeeper.instance().get_librarian(book_id)
    return librarian.ask_question_logged(question)


@app.route("/api/books/<book_id>/history", methods=["GET"])
def history(book_id):
    return BookKeeper.instance().list_chat_logs(book_id)


@app.route("/api/books/<book_id>/history/<log_id>", methods=["DELETE"])
def remove_history(book_id, log_id):
    BookKeeper.instance().remove_chat_log(book_id, log_id)
    return "OK"


@app.route("/<path:static_file>")
def serve_static_file(static_file):
    return app.send_static_file(static_file)


@app.errorhandler(ValueError)
def handle_exception(e):
    response = jsonify(
        {"error": {"type": e.__class__.__name__, "message": str(e)}}
    )
    response.status_code = 400

    return response


@app.errorhandler(AuthenticationError)
def handle_exception(e):
    response = jsonify(
        {"error": {"type": e.__class__.__name__, "message": str(e)}}
    )
    response.status_code = 400

    return response
