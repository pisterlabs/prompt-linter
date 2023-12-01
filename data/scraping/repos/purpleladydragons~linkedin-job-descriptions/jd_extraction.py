import os
import openai

openai.api_key = os.environ['OPENAI_API_KEY']

def analyze_job_description(description):
    print('Prompting...')
    prompt = f"What tools and languages are mentioned in this job description? Please just list them in a comma-separated list.\n\n{description}"
    response = openai.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {'role': 'user', 'content': prompt}
        ]
    )

    return response.choices[0].message.content.strip()

