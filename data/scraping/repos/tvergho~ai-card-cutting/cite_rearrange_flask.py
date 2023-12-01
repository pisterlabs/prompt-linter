import os
import openai
from flask import Flask, request
from dotenv import load_dotenv

load_dotenv()
app = Flask(__name__)
openai.api_key = os.getenv("OPENAI_API_KEY")

def add_example(prompt, example):
    prompt += "\n\nExample\n\"\"\""
    prompt += example
    prompt += "\"\"\""
    return prompt

def generate_prompt(citation):
  prompt = "Rearrange the citation to match the format of the examples. Do not output any info in the final result which is not in the Provided Citation, including the month and date."
  prompt = add_example(prompt, "\n**Talki ’21** [Valentina; Professor of Law; 2021; “International Legal Personality of Artificial Intelligence”; https://doi.org/10.3390/laws10040085; Laws, Vol. 10, No. 4]")
  prompt = add_example(prompt, "\n**Kerry et al. ’21** [Cameron F. Kerry, Joshua P. Meltzer, Andrea Renda; Distinguished Fellow, Governance Studies, Center for Technology Innovation; Senior Fellow at Brookings; Senior Research Fellow and Head of Global Governance; 10/25/21; ”Strengthening Global Governance”; https://www.brookings.edu/research/strengthening-cooperation-on-ai; Brookings Institution]")
  prompt = add_example(prompt, "\n**Maas ’21** [Matthijs M; April 2021; https://matthijsmaas.com/uploads/Maas; University of Copenhagen]")
  prompt = add_example(prompt, "\nTzimas ‘19 [Themistoklis; PhD in international law; “The Need for an International Treaty for AI”; 2019; https://ideas.repec.org/v4y2019i1p73-91.html; Scientia Moralitas, Vol. 4]")
  prompt += f"\n\Provided Citation\n\"\"\"\n{citation}\n\"\"\"\n\nRearranged Citation"
  return prompt

@app.route("/")
def get_competion():
  citation = request.args.get('citation', '')
  # prompt = f"Rearrange the citation to match the format of the examples. Do not output any info in the final result that is not provided in the citation.\n\nExample 1\n\"\"\"\n**Talki ’21** [Valentina; Professor of Law; 2021; “International Legal Personality of Artificial Intelligence”; https://doi.org/10.3390/laws10040085; Laws, Vol. 10, No. 4]\n\"\"\"\n\nExample 2\n\"\"\"\n**Kerry et al. ’21** [Cameron F. Kerry, Joshua P. Meltzer, Andrea Renda; Distinguished Fellow, Governance Studies, Center for Technology Innovation; Senior Fellow at Brookings; Senior Research Fellow and Head of Global Governance; 10/25/21; ”Strengthening Global Governance”; https://www.brookings.edu/research/strengthening-cooperation-on-ai/; Brookings Institution]\n\"\"\"\n\nExample 3\n\"\"\"\n**Maas ’21** [Matthijs M; April 2021; https://matthijsmaas.com/uploads/Maas; University of Copenhagen]\n\"\"\"\n\nCitation\n\"\"\"\n{citation}\n\"\"\"\n\nRearranged Citation"
  prompt = generate_prompt(citation)
  response = openai.Completion.create(
    model="text-davinci-003",
    prompt=prompt,
    temperature=0,
    max_tokens=int((len(citation)/4) + 50),
    top_p=1,
    frequency_penalty=0,
    presence_penalty=0
  )

  if response.choices:
    # Strip newline characters
    return response.choices[0].text
  else:
    return ""