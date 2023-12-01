import re
import PyPDF2
import nltk
from nltk.tokenize import sent_tokenize
import openai
from azure.ai.textanalytics import TextAnalyticsClient
from azure.core.credentials import AzureKeyCredential
from nltk.tokenize import sent_tokenize

# Set your Azure Text Analytics API key and endpoint
# azure_key = 'your-azure-key'
# azure_endpoint = 'your-azure-endpoint'

# # Create a Text Analytics client
# credential = AzureKeyCredential(azure_key)
# text_analytics_client = TextAnalyticsClient(endpoint=azure_endpoint, credential=credential)


openai.api_key = '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'

def extract_cgpa(text):
    column_name_pattern = re.compile(r'GPA\s*/\s*Marks\s*\(%\)', re.IGNORECASE)
    match_column = column_name_pattern.search(text)

    if match_column:
        # Look for the value after the column name
        value_pattern = re.compile(r'\b\d+(\.\d+)?\b')
        match_value = value_pattern.search(text[match_column.end():])

        if match_value:
            return match_value.group()
    return None

def extract_project_summary(text):
    projects_pattern = re.compile(r'PROJECTS(.*?)(?=(TECHNICAL SKILLS|$))', re.DOTALL | re.IGNORECASE)
    match = projects_pattern.search(text)

    if match:
        projects_text = match.group(1).strip()
        return projects_text

    return None

def generate_detailed_summary(project_details):
    prompt = f"Summarize the following projects:\n{project_details}\n\nSummary:"
    response = openai.Completion.create(
        engine="text-davinci-003",  # You may need to check the latest available engine
        prompt=prompt,
        temperature=0.7,
        max_tokens=200,
        n=1,
    )
    generated_text = response.choices[0].text.strip()
    return generated_text



def extract_text_from_pdf(pdf_path):
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        text = ''
        for page_num in range(len(reader.pages)):
            text += reader.pages[page_num].extract_text()

        # print(text)
    return text


def extract_candidate_name(text):
    # Extract the candidate name as the first string before a newline character
    lines = text.split('\n')
    candidate_name = lines[0].strip() if lines else None
    return candidate_name


def extract_project_summary(text):
    #working
    # Extract projects section based on the specific label "PROJECTS"
    # projects_pattern = re.compile(r'PROJECTS(.*?)(?=(PROJECTS|$))', re.DOTALL | re.IGNORECASE)
    projects_pattern = re.compile(r'PROJECTS(.*?)(?=(TECHNICAL SKILLS|$))', re.DOTALL | re.IGNORECASE)
    match = projects_pattern.search(text)

    if match:
        projects_text = match.group(1).strip()
        # print(projects_text)
        # Split projects based on bullet points
        projects_list = re.split(r'\s*•\s*', projects_text)

        # Use sentence summarization (extract the first sentence as a summary) for each project
        summaries = []
        for project in projects_list:
            if project:
                # Split the project details based on hyphens
                details = re.split(r'\s*-\s*', project)
                # Use the first sentence of the first detail as a summary
                summary = sent_tokenize(details[0])[0] if details else None
                summaries.append(summary)

        # Combine project summaries into a single paragraph
        summary_paragraph = ' '.join(summaries)

        return summary_paragraph if summaries else None

    return None


# def generate_detailed_summary(project_details):
#     prompt = f"Summarize the following projects:\n{project_details}\n\nSummary:"
#     response = openai.Completion.create(
#         engine="text-davinci-003",  # You may need to check the latest available engine
#         prompt=prompt,
#         temperature=0.7,
#         max_tokens=200,
#         n=1,
#     )
#     generated_text = response.choices[0].text.strip()
#     return generated_text

def generate_detailed_summary_azure(project_details):
    document = [
        {"id": "1", "language": "en", "text": project_details}
    ]
    
    response = text_analytics_client.analyze_sentiment(inputs=document)
    
    # Extracting the sentiment score for a basic summary
    sentiment_score = response[0].confidence_scores.positive
    
    if sentiment_score >= 0.7:
        return "Positive sentiment. The projects are generally well-received."
    elif sentiment_score >= 0.4:
        return "Neutral sentiment. The projects have mixed feedback."
    else:
        return "Negative sentiment. Some concerns may be present in the projects."



def nextract_project_summary(text):
    # Extract projects section based on the specific label "PROJECTS"
    projects_pattern = re.compile(r'PROJECTS(.*?)(?=(?:[A-Z ]+|$))', re.DOTALL | re.IGNORECASE)
    match = projects_pattern.search(text)

    if match:
        projects_text = match.group(1).strip()

        # Split projects based on bullet points
        projects_list = re.split(r'\s*•\s*', projects_text)

        # Use sentence summarization (extract the first sentence as a summary) for each project
        summaries = []
        for project in projects_list:
            if project:
                # Split the project details based on hyphens
                details = re.split(r'\s*-\s*', project)
                # Use the first sentence of the first detail as a summary
                summary = sent_tokenize(details[0])[0] if details else None
                summaries.append(summary)

        # Combine project summaries into a single paragraph
        summary_paragraph = ' '.join(summaries)

        return summary_paragraph if summaries else None

    return None


def oextract_project_summary(text):
    # Extract projects section based on a pattern (you might need to adjust this)
    projects_pattern = re.compile(r'•\s*([^:]*:.*?)(?=(•|$))', re.DOTALL | re.IGNORECASE)
    matches = projects_pattern.findall(text)

    summaries = []
    for match in matches:
        project_details = match.strip()
        # Split project details into project name and description based on colon
        project_parts = re.split(r'\s*:\s*', project_details, maxsplit=1)
        if len(project_parts) == 2:
            project_name, project_description = project_parts
            # Use sentence summarization (extract the first sentence as a summary) for each project
            summary = sent_tokenize(project_description)[0] if project_description else None
            summaries.append(f"{project_name}: {summary}")

    # Combine project summaries into a single paragraph
    summary_paragraph = '\n'.join(summaries)

    return summary_paragraph if summaries else None


def pextract_project_summary(text):
    # Extract projects section based on a pattern (you might need to adjust this)
    projects_pattern = re.compile(r'\bPROJECTS\b(.*?)(?=\b[A-Z])', re.DOTALL | re.IGNORECASE)
    match = projects_pattern.search(text)
    # print(text)
    if match:
        projects_text = match.group(1).strip()
        # print(projects_text)
        # Split projects based on bullet points
        projects_list = re.split(r'\s*•\s*', projects_text)
        # print(projects_list)
        # Use sentence summarization (extract the first sentence as a summary) for each project
        summaries = []
        for project in projects_list:
            if project:
                # Split the project details based on hyphens
                details = re.split(r'\s*-\s*', project)
                # Use the first sentence of the first detail as a summary
                summary = sent_tokenize(details[0])[0] if details else None
                summaries.append(summary)

        # Combine summaries into a single paragraph
        summary_paragraph = ' '.join(summaries)

        return summary_paragraph

    return None
