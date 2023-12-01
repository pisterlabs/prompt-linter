import openai
from utils import parse_results

def generate_syllabus(chat_completion: openai.ChatCompletion, text_body: str) -> str:
    model_output = chat_completion.create(
        model = "gpt-3.5-turbo",
        messages = [
            {"role": "system", "content":f"Your job is to read the information and extract the Grade Weightages, Office Hour timings, whether or not the course has mandatory attendance, the late submission policy and the regrading policy and output it in the given format:\n\n - Grade weightages/Breakdown: \n  - Component 1 - X% \n   - Component 2 - X% \n Office Hours: \n Mandatory Attendance: <Yes/No>\n Late submissing policy: \n Regrading policy: "},
            {"role": "user", "content":f"{text_body}"}
            ]
        )
    output = parse_results(model_output)
    return output
