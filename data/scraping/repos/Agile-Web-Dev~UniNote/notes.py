from flask import current_app, make_response, request, session
from sqlalchemy import desc

from app import db
from app.chat.utils import openai
from app.models import Note

from . import bp


@bp.route("/<class_id>", methods=["GET"])
def get_notes(class_id):
    """get notes based on the classID"""
    res_arr = []
    res = (
        db.session.query(Note)
        .filter(Note.class_id == class_id)
        .order_by(desc(Note.created_at))
    )
    for entry in res:
        res_arr.append(entry.serialize())
    return res_arr


# when user saves their notes
@bp.route("", methods=["POST"])
def post_notes():
    """
    upload notes based on the classID into database
    endpoint: /api/class/notes
    """

    data = request.get_json()

    created_by = data.get("createdBy", "")
    class_id = data.get("classId", "")
    title = data.get("title", "")
    content = data.get("content", "")

    note = Note(created_by=created_by, class_id=class_id, title=title, content=content)
    db.session.add(note)
    db.session.commit()

    result = note.serialize()

    if not current_app.config["TESTING"]:
        openai.add_document(session["class_id"], title + "\n\n" + content)

    if not isinstance(result, dict):
        return make_response({"msg": "Internal server error"}, 500)

    return make_response(result, 201)
