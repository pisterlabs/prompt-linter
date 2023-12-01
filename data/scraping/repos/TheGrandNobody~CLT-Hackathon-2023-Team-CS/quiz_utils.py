import os
import json
from io import BytesIO
import openai
import tempfile
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.pdfgen import canvas
from typing import Dict, List, Tuple, Optional
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

def validate_question_format(question_list: List) -> Tuple[bool, str]:
    # Check if input is a list
    if not isinstance(question_list, list):
        return (False, "Input is not a list.")
    
    for i, question_dict in enumerate(question_list):
        # Check if each entry is a dictionary
        if not isinstance(question_dict, dict):
            return (False, f"Element at index {i} is not a dictionary.")
        
        # Check if all required keys are present
        if not all(key in question_dict for key in ("q", "options", "a_index")):
            return (False, f"Element at index {i} is missing one or more required keys ('q', 'options', 'a_index').")
        
        # Check if 'q' is a string
        if not isinstance(question_dict["q"], str):
            return (False, f"Element at index {i} has a 'q' value that is not a string.")
        
        # Check if 'options' is a list of strings
        if not isinstance(question_dict["options"], list):
            return (False, f"Element at index {i} has an 'options' value that is not a list.")
        
        if not all(isinstance(option, str) for option in question_dict["options"]):
            return (False, f"Element at index {i} has an 'options' value that contains non-string elements.")
        
        # Check if 'a_index' is an integer and within the range of options
        if not isinstance(question_dict["a_index"], int):
            return (False, f"Element at index {i} has an 'a_index' value that is not an integer.")
        
        if question_dict["a_index"] < 0 or question_dict["a_index"] >= len(question_dict["options"]):
            return (False, f"Element at index {i} has an 'a_index' value that is out of range for the given options.")
    
    return (True, "")

EXAMPLE_QUIZ = [
    {
        "q": "What is the capital of France?", 
        "options": ["Paris", "Berlin", "London", "Madrid"],
        "a_index": 0
    },
    {
        "q": "Which of the following is not a programming language?", 
        "options": ["Python", "Java", "Banana", "TypeScript"],
        "a_index": 2
    },
    {
        "q": "What is 2+2?", 
        "options": ["34", "5", "15", "4"],
        "a_index": 3
    }
]


def get_quiz_for_prompt(messages: List[Dict], openai_api_key: str, max_tries: int = 4) -> Tuple[Optional[Dict], List[Dict]]:
    """
    Prompts chatgpt until a proper json object representing a quiz is returned.

    returns: tuple (quiz_dict, new_messages)
    """
    openai.api_key = openai_api_key

    history = messages.copy()
    next_prompt = ""
    for i in range(max_tries):
        print(f"getting quiz from chatgpt attempt {i+1}/{max_tries}")
        #st.session_state.messages.append({"role": "user", "content": f"Give me the quiz with the answers as well and make it multiple choice please"})
        if not next_prompt:
            next_prompt = f"Give me the quiz on the topic above, including the answers. You're response must consist only of a list of valid json objects (starting with the char '[' and ending with ']') of a format like the following example valid response:\n{EXAMPLE_QUIZ}"
        print("next_prompt = ", next_prompt)
        history.append({"role": "user", "content": next_prompt})
        response = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=history)
        msg = response.choices[0].message

        if "[" not in msg.content or "]" not in msg.content:
            next_prompt = "response does not contain '[' or ']'"
            continue
        raw_answer = msg.content[msg.content.index("["):msg.content.rindex("]")+1]

        # try to read as json and validate
        print("\nresponse = ", raw_answer)
        try:
            quiz_dict = json.loads(raw_answer)
            result, problem = validate_question_format(quiz_dict)
            if result:
                return (quiz_dict, history)
            history.append({"role": "assistant", "content": f"Invalid quiz format please try again: {problem}"})
        except json.decoder.JSONDecodeError as err:
            next_prompt = f"invalid json provided please try again. Respond only with valid json.\n{err}"
    
    print(f"Failed to get quiz from chatgpt after {max_tries} attempts :(")
    return (None, history)

def generate_pdf(quiz_data: Dict, fname: Optional[str] = None):
    """Conver quiz dictionary object to a pdf file (written to disk)."""
    buffer = BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter

    y_position = height - 50

    # Generate questions
    for i, question in enumerate(quiz_data):
        y_position -= 50
        c.drawString(100, y_position, f"{i + 1}. {question['q']}")
        
        for j, option in enumerate(question['options']):
            y_position -= 20
            c.drawString(120, y_position, f"{chr(65+j)}. {option}")
            
        # Check for end of page
        if y_position < 50:
            c.showPage()
            y_position = height - 50

    # Create a new page for answers
    c.showPage()
    y_position = height - 50
    c.drawString(100, y_position, "Answers:")

    for i, question in enumerate(quiz_data):
        y_position -= 20
        #correct_answer = question['options'][question['a_index']]
        correct_answer = chr(ord("A") + question['a_index'])
        c.drawString(120, y_position, f"{i + 1}. {correct_answer}")

    c.save()
    pdf_data = buffer.getvalue()
    buffer.close()

    if fname is not None:
        print(f"writing pdf to {fname}")
        with open(fname, "wb") as f:
            f.write(pdf_data)
    return pdf_data

if __name__ == "__main__":
    result, message = validate_question_format(EXAMPLE_QUIZ)
    print(result)  # Should print True
    print(message)  # Should print ""
    assert result == True
    assert message == ""

    prefix = os.path.join(SCRIPT_DIR, ".temp")

    # test writing pdf
    temp_dir_obj = tempfile.TemporaryDirectory(
        prefix=prefix,
    )
    work_dir = temp_dir_obj.name
    print(f"using workDir = '{work_dir}'")
    fname = os.path.join(work_dir, "quiz.pdf")

    generate_pdf(EXAMPLE_QUIZ, fname)
    import pdb; pdb.set_trace()
