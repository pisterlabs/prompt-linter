# output.py

import openai
import os

# Set OpenAI API key
openai.api_key = os.environ['OPENAI_API_KEY']

def generate_response(prompt, relevant_cases, case_infos):
    case_info = ", ".join([f"{case['name_abbreviation']}: {case['citation']}: {case['helpful_parenthetical']}" for case in case_infos])  # Add the URL here
    
    print("Case info:", case_info)  # Add this line to print case_infos
    
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=f"Based only on the case information in: '{case_info}', please provide a concise answer to the following legal question: '{prompt}'. Please be sure to cite case law which supports each legal proposition in your answer. Ignore any off-topic or irrelevant information in 'case_infos'. Finally, provide a straightforward legal summary of the law as it applies to the question.",
        max_tokens=2000,
        temperature=0.5,
        top_p=1.0,
        frequency_penalty=0.2,
        presence_penalty=0.2
    )

    answer = response.choices[0].text.strip()

    # Post-process the answer to add hyperlink HTML tags around the case names
    for case in case_infos:
        case_name = case['name_abbreviation']
        case_url = case['html_url']
        answer = answer.replace(case_name, f'<a style="color:#dfd" href="{case_url}" target="_blank">{case_name}</a>')
        answer = answer.replace('\n', '<br>')

    # Generate case summaries
    summarize_buttons = [{'name_abbreviation': case['name_abbreviation'], 'url': case['url']} for case in case_infos]


    print("Generated answer:", answer)

    return answer, summarize_buttons
