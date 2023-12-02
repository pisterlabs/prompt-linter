import openai


def analyse_cv(job_description, skills, resume_text):
    print("CV Analyzer Started")
    user_message = [
        {"role": "system", "content": """You are a HR. You want to hire candidates for your company. you get many resume for that job description Your job is to rank the resumes according to their alignment with the job title, skills, and resumes texts. 
                Input will be in the following format - 
                job title - job title will be provided here
                skills - skills will be provided here
                resume text - resume text will be provided here
                Output should be in the form of json object with the following keys - 
                score :  the score of the resume out of 10 based on job title, skills, and resume text
                """},
        {'role': 'user', 'content': f"""job description  - {job_description} skills -{skills} and resume text - {resume_text}"""}
    ]

    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=user_message
    )

    print("CV Analyzer Completed")
    return response['choices'][0]['message']['content']
