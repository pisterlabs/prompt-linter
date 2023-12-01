from langchain.llms import Ollama
import json
from gpa import extract_text_pdf

model = Ollama(base_url="http://localhost:11434", model="mistral", temperature=0.75)


def extract_education(text):
    prompt = """
You are given a text extracted from resume of an individual. 
You need to analyze the text to the find the one latest education qualification of the individual mentioned, 
Identify the degree and the major of the degree in the below format:
{
    "Degree":"[Identified Degree]",
    "Major":"[Identified Major]"
}

If no degree is found, return 'Not Found'
"""
    query = """
INSTRUCTION = Analyze the text and strictly the output in the mentioned format. 
Prompt = {}
Text = {}
""".format(
        prompt, text
    )
    result = model(query)
    output = json.loads(result)
    return output["Major"]


print(extract_education(extract_text_pdf('./resume/resume/ShivNadarIOE_MrinalTiwari_2010110404_Resume.pdf')))
print(extract_education(extract_text_pdf('./resume/resume/Professional CV Resume.pdf')))

# why not get the full degree and major together
