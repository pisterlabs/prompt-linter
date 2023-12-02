from langchain.llms import Ollama
import re
from pdfminer.high_level import extract_text

model = Ollama(base_url="http://localhost:11434", model="llama2", temperature=0.1)

def extract_text_pdf(file_path):
    return extract_text(file_path)


def find_gpa_near_education(text):
    # Regular expression pattern for education-related keywords
    education_keywords = r'\b(University|College|Bachelor|Master|B\.?Tech|M\.?Tech|PhD|BA|BS|MA|MS|BSc|MSc)\b'
    # Regular expression pattern for GPA-like numbers (e.g., 3.5, 4.0, 85, 92.5)
    gpa_pattern = r'(?:GPA[:\s-]*)?([0-9]\.[0-9]{1,2}|10(?:\.0{1,2})?)'

    for match in re.finditer(education_keywords, text):
        # Check in the vicinity of the education keyword for a GPA-like number
        start, end = max(0, match.start() - 100), match.end() + 100
        nearby_text = text[start:end]
        gpa_match = re.search(gpa_pattern, nearby_text)
        if gpa_match:
            return gpa_match.group(0)
    return "Not found"


def extract_gpa(text):
    # to-do : convert to gpa out of 4 
    text = find_gpa_near_education(text)
    prompt = """
    Given the following text, please extract the numerical value of the GPA. 
    The text may contain information related to education, and the GPA might be mentioned 
    in various formats. Your task is to identify 
    and extract the numerical value associated with the GPA.
    """
    query = """
    INSTRUCTION = Generate only the final numerical value without a sentence. Do not provide any explanation or additional information.
    PROMPT = {}
    Text = {}""".format(prompt,text)

    result = float(model(query).strip())
    return result

