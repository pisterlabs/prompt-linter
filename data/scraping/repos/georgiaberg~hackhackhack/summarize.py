from flask import request, jsonify
from flask_restful import Resource
import cohere
import re
from datetime import datetime



def getSummary(notes_string, format="paragraph"):
    """Summarize an array of notes using Cohere's summarize API."""

    # Use Cohere API to summarize the notes. If they are too short, use the generate API instead.
    try:
        response = co.generate(
            prompt=f"Summarize the following journal entries coherently in < 250 words with second-person, reflection-inspiring tone.\n{notes_string}",
            max_tokens=250,
            temperature=0.3,
            model="command"
        )
        summary = response.generations[0].text.strip()
    except:
        response = co.summarize(
            text=notes_string,
            format=format,
            additional_command="in a second-person, reflection-inspiring tone",
        )
        summary = response.summary
    return summary


# categories to leverage: experience, to-do, relationship, goal
def getQuestions(notes_string):
    """Generate reflective questions & exercises based on notes using Cohere's generate API."""
    prompt = f"""Based on the following collection of notes, generate reflective questions or exercises that could help deepen understanding or inspire reflection. The notes are as follows:

        {notes_string}

        Please generate up to 5 reflective questions or writing exercises. Make sure to cite the title and date of the note that each question or exercise relates to.
        If that's not possible, omit the title and date. Follow this format and don't include extra content after the last question or exercise.
        Example output:
        1. How effectively were tasks delegated in the 'Meeting with Team' note? (Meeting with Team, 2023-09-16)
        2. What aspects of the code could be improved based on your 'Code Review' note? (Code Review, 2023-09-15)
        3. How did the visit to the zoo with Aunt Madalyn make you feel? (Client Feedback, 2023-09-14)*
        """

    questions_response = co.generate(prompt=prompt, max_tokens=175, temperature=0.0, model="command-light", end_sequences=["*"])
    content = questions_response.generations[0].text

    split_text = content.split("\n")
    # remove the numbers from each question
    split_text = [re.sub(r"^\d+\.", "", q.strip()) for q in split_text]
    questions = [q.strip() for q in split_text]
    return questions


class SummarizeNotes(Resource):
    """Summarize notes using Cohere API."""

    def post(self):
        notes = request.json["notes"]
        notes_string = ""

        for note in notes:
            try:
                notes_string += note["title"] + ", Date:" + note["date"] + "\n"
                notes_string += "Sentiment:" + note["sentiment"] + "\n"
                notes_string += "Types:" + ",".join(note["types"]) + "\n"
            except:
                pass
            finally:
                notes_string += note["content"] + "\n"

        format = request.json.get("format", "paragraph")
        summary = getSummary(notes_string, format)
        questions = getQuestions(notes_string)
        return jsonify({"summary": summary, "questions": questions})
