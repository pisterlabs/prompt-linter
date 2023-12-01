import openai
import re
import default_prompts
# Set up OpenAI API credentials


# Define function to summarize an article based on a prompt

def ask(query):
    # Use OpenAI's GPT-3 API to generate a summary
    response = openai.Completion.create(
      engine="text-davinci-002",
      prompt=query,
      temperature=0.5,
      max_tokens=1024,
      top_p=1,
      frequency_penalty=0,
      presence_penalty=0
    )
    res = response.choices[0].text
    res = res.strip()
    res = re.sub(r'[\r\n]+', ' ', res)
    return res

# 新闻摘要
def summarize_article(input_content, prompt=default_prompts.Commentariat):
    text = prompt + '\n\n' +'please summary and give opinion on this article:' + input_content
    return ask(text)

# 翻译
def translate2english(input_content, prompt=default_prompts.Improver):
    text = prompt + '\n\n' +'please translate to English:' + input_content
    return ask(text)

# 手稿润色
def polish_manuscript(input_content, prompt=default_prompts.Researcher):
    text = prompt + '\n\n' +'please polish my manuscript:' + input_content
    return ask(text)

# 影评
def movie_critic(input_content, prompt=default_prompts.Researcher):
    text = prompt + '\n\n' +'write a movie review for the movie' + input_content
    return ask(text)


# 招聘信息撰写
def write_job_posting(requirements, responsibilities, prompt=default_prompts.Researcher):
    # Write a compelling job title
    job_title = "Software Engineer"
    
    # Write a brief summary of the job
    job_summary = "We are seeking a talented software engineer to join our team and help develop cutting-edge software solutions."
    
    # Write the job requirements
    job_requirements = "Requirements:\n" + requirements
    
    # Write the job responsibilities
    job_responsibilities = "Responsibilities:\n" + responsibilities
    
    # Concatenate all the job details and return the job posting
    job_posting = job_title + "\n\n" + job_summary + "\n\n" + job_requirements + "\n\n" + job_responsibilities

    text = prompt + '\n\n' +'write a job hunting description:' + job_posting
    return ask(text)


# 应聘信息, CV撰写
def write_cv(name, email, phone, address, work_experience, education, prompt=default_prompts.Improver):
    # Format the contact details
    contact_details = f"{name}\n{email}\n{phone}\n{address}\n\n"
    
    # Format the work experience section
    work_experience_section = "Work Experience:\n\n"
    for experience in work_experience:
        work_experience_section += f"{experience['position']}\n{experience['company']}\n{experience['dates']}\n{experience['description']}\n\n"
    
    # Format the education section
    education_section = "Education:\n\n"
    for degree in education:
        education_section += f"{degree['degree']}\n{degree['institution']}\n{degree['dates']}\n{degree['description']}\n\n"
    
    # Concatenate all the sections and return the CV
    cv = contact_details + work_experience_section + education_section

    text = prompt + "\n\n" +  'write a formal cv given the following decription:' + cv
    return ask(text)

# 套瓷信
def write_cover_letter(to_who, name, job_title, company_name, job_description, prompt=default_prompts.Researcher):
    # Write the introduction
    intro = f"Dear {to_who},\n\nI am writing to express my interest in the {job_title} position at {company_name}."

    # Write the body of the cover letter
    body = f"I believe that my skills and experience make me a strong candidate for the position. {job_description}\n\nI look forward to the opportunity to discuss my qualifications further."

    # Write the closing
    closing = "\n\nSincerely,\n\n" + name

    # Concatenate all the sections and return the cover letter
    cover_letter = intro + "\n\n" + body + closing

    text = prompt + "\n\n" +  'write a formal cover letter given the following decription:' + cover_letter
    return ask(text)

# 请假信
def generate_leave_of_absence_prompt(reason, recipient, prompt=default_prompts.Improver):
    """
    Generates a prompt for ChatGPT to continue the text generation of a leave of absence letter.
    """
    text =  prompt + '\n\n' + f"Write a leave of absence letter to {recipient} requesting a leave of absence starting on [start date] until [end date]. The reason for the absence is {reason}."
    return ask(text)


# 科学问题
def response_scientific_question(question, prompt=default_prompts.Scientist):
    text =  prompt + '\n\n' + f" answer my question and give some explainations and suggestions, my question is: {question}"
    return ask(text)
