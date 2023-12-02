import re
import time
from typing import List, Dict

import openai
import PyPDF2

from .chat_utils import call_openai


def analysis(interview_so_far: List[Dict[str, str]]):
    roles = ["Interviewer", "Interviewee"]
    interview = "\n\n".join(f"{roles[i % 2]}: {message['content']}" for i, message in enumerate(interview_so_far))

    categories = {
        "Experience": {
            "description": "Evaluate the interviewee's past work experiences and projects. Focus on understanding the depth and breadth of their involvement, the challenges they faced, and how they overcame them. Assess how relevant their previous work is to the role they are interviewing for.",
            "metric": "Rate the interviewee's experience relevance to the job role on a scale from 1 to 10, where 1 is not relevant at all, and 10 is highly relevant."
        },
        "Problem Solving": {
            "description": "Evaluate the interviewee's problem-solving skills and their approach to tackling challenges. Use hypothetical scenarios related to the job role to understand their thought process and decision-making abilities.",
            "metric": "Score the interviewee's problem-solving skills on a scale from 1 to 10, based on their ability to logically approach and solve the presented scenarios."
        },
        "Personality": {
            "description": "Gauge the interviewee's personality traits, including their emotional intelligence, resilience, and their ability to work under pressure. Understand their interpersonal skills and how they relate to colleagues and clients.",
            "metric": "Evaluate the interviewee's personality fit for the team and role on a scale from 1 to 10, taking into consideration the interpersonal dynamics of the existing team."
        },
        "Learning and Growth": {
            "description": "Assess the interviewee's willingness and ability to learn and grow within the role. Understand their approach to professional development and continuous learning.",
            "metric": "Rate the interviewee's potential for growth and development in the role on a scale from 1 to 10, where 1 indicates low potential, and 10 indicates high potential."
        },
        "Communication": {
            "description": "Assess the interviewee's communication skills, including their clarity of expression, listening skills, and their ability to articulate their thoughts effectively.",
            "metric": "Provide a rating for the interviewee's communication skills on a scale from 1 to 10, where 1 indicates poor communication, and 10 indicates excellent communication."
        },
        "Leadership": {
            "description": "If applicable, assess the interviewee's leadership skills and their experience in managing or leading teams. Understand their approach to leadership, decision-making, and their ability to inspire and motivate others.",
            "metric": "Rate the interviewee's leadership skills on a scale from 1 to 10 taking into account their past leadership experiences and their demonstrated ability to lead."
        },
        "Cultural Fit": {
            "description": "Evaluate how well the interviewee would fit into the company’s culture. Discuss their values, work style, and expectations to understand how they align with the company's environment.",
            "metric": "Rate the interviewee's cultural fit on a scale from 1 to 10, where 1 is not a fit at all, and 10 is a perfect fit."
        }
    }
    
    for category, info in categories.items():
        prompt = (
            f"Please briefly evaluate the interviewee's {category.lower()} based on their responses.\n\n"
            f"Category description: {info['description']}\n\n"
            f"Category metric: {info['metric']} (1-10)\n\n"
            f"```\n{interview}\n```\n"
            f"IMPORTANT: Give your evaluation in the form of a json object with only keys 'score' and 'comment'.\n\n"
        )
        
        response_content = call_openai(prompt)
        if response_content:
            match = re.search(r'"score":\s*(\d+),\s*"comment":\s*"([^"]+)"', response_content)
            if match:
                score, comment = match.groups()
                categories[category]['score'] = score
                categories[category]['comment'] = comment
                print(f"{category}: Score - {score}, Comment - {comment}")
            else:
                print(f"Failed to extract score and comment for {category}")
        else:
            print(f"Failed to get response for {category}")

    return categories


def extract_resume(path): 
    with open(path, 'rb') as file: 
        PDF = PyPDF2.PdfReader(file)
        pages = len(PDF.pages)
        key = '/Annots'
        uri = '/URI'
        ank = '/A'

        text = [] 

        for page in range(pages):
            pageSliced = PDF.pages[page]
            text += [pageSliced.extract_text()]

        return text


def resume_analyzer(file_path):
    """
    analyze resume from file_path
    """
    pass


def get_feedback(model, company_name, job_description, text_resume): 
    start_time = time.time()

    prompt = f"""
    Given below is a job description and the resume of the applicant who is applying for the position. They are labelled as {{DESCRIPTION}} and {{RESUME}}.

    ```
    {{DESCRIPTION}}
    {job_description}
    ```

    ```
    {{RESUME}}
    {text_resume}
    ```

    Read the above carefully, and produce a relatively detailed analysis of bullet points and comments on the fit between the resume and the job. Make critical comments and kind comments if possible.
    """

    response = openai.ChatCompletion.create(
        model=model,
        messages=[
            {"role": "system", "content": f"You are an assistant that is helping a company, {company_name}, generate interview questions and assess the candidacy of applicants. You must evaluate all resumes with a focus on the overall skills listed in each of them."},
            {"role": "user", "content": prompt}, 
        ],
    )

    generated_text = response['choices'][0]['message']['content'].strip()

    print(model, ":", time.time() - start_time)

    return generated_text

