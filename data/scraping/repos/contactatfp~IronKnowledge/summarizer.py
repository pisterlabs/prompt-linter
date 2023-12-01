import openai
from project import get_project_details
from config import Config

openai.api_key = Config.GPT4_API_KEY


def load_gpt4_model():
    model_name = "gpt-4"
    return model_name


def generate_summary(model_name, input_data):
    response = openai.Completion.create(
        engine=model_name,
        max_tokens=1970,
        temperature=0,
        top_p=1,
        n=1,
        stop=None,
        prompt=input_data
    )
    summary = response.choices[0].text
    return summary.strip()


def save_summary(project_id, summary):
    project = get_project_details(project_id)

    # Save the summary to the living document
    if project:
        project['summary'] = summary

    return True
