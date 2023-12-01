from claude_api import Client
import glob
import re
import tqdm

import anthropic
from PyPDF2 import PdfReader

claude_key = ''
client = anthropic.Client(claude_key)

sections = [
'Rehabilitative and Habilitative Services and Devices', 'Outpatient Services', 'Other Services', 'Pediatric Dental Care',
    'Preventive and Wellness Services and Chronic Disease Management', 'Mental Health and Substance Abuse Services',
    'Prescription Drugs', 'Pediatric Vision', 'Hospitalization', 'Emergency Services',
    'Laboratory Outpatient and Professional Services'
]

prompt = """
Give me all the {}, split into a paragraph with the following structure:

{Page of the document this is contained}{

{Benefit}: {Name of Benefit}

{In network Cost Share}: {Value}

{Out of Network Cost Share}: {Value}

{Description}: {Value}}

Create a new paragraph for each new benefit and separate them with a new line. Be specific with the pages and where the data is organized. If it is split in separate pages, return both of them in the page index. The document is 8 pages long.
"""

path_to_pdfs = '/Users/mariadobko/Downloads/Plans pdfs/**'
for pdf_file in glob.glob(path_to_pdfs):
    pdf_file_name = pdf_file.split('/')[-1].split('.')[0]
    reader = PdfReader(pdf_file)
    context_prompt = ''
    for page in reader.pages:
        context_prompt += page.extract_text()

    result = ''
    pages = {k: [] for k in sections}
    for k in tqdm.tqdm(sections):
        user_query = prompt.replace('{}', k)
        summary_prompt = f"{anthropic.HUMAN_PROMPT} {context_prompt}. My query is {user_query}. {anthropic.AI_PROMPT}"

        response = client.completion(
            prompt=summary_prompt,
            stop_sequences=[anthropic.HUMAN_PROMPT],
            model="claude-v1-100k",
            max_tokens_to_sample=5000,
        )
        response_res = response['completion']
        pages_section = [x[:2] for x in response_res.split('Page')]
        for i in pages_section:
            try:
                pages[k].append(int(i))
            except:
                pass
        response_res = ['{' + x.replace('\n', '') for x in response_res.split('{')]
        response_res = '\n'.join([x for x in response_res if ('{Page' not in x) and (x != '{') and len(x)> 12])
        response_res = '\n'.join([x + ' Pages: {}'.format(str(set(pages[k]))) for x in response_res.split('\n')])
        result += response_res + '\n'

    final_result = []
    for i in result:
        if len(i) > 10:
            final_result.append(i)

    with open('{}.txt'.format(pdf_file_name), 'w') as f:
        f.write(result)
