from dotenv import load_dotenv
import os
import shutil

from langchain.document_loaders import PDFPlumberLoader
from langchain.chat_models import ChatOpenAI
from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
)

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")


def ingest_pdf(resume):
    print("Loading resume...")
    loader = PDFPlumberLoader(resume)
    documents = loader.load()
    resume_text = " ".join(document.page_content for document in documents)
    return resume_text


def get_score(resume_text):
    print("Getting score...")
    llm = ChatOpenAI(
        model="gpt-3.5-turbo-16k", temperature=0.0, openai_api_key=openai_api_key
    )
    template = f"""\
You are an Industrial-Organizational Psychologist who specializes in personnel selection and assessment. 
Your discipline of study, Industrial-Organizational Psychology, would best prepare you to answer the 
question or perform the task of determining a job fit score based on a resume and a job description. 

You will review the following resume and job description and determine a job fit score as a float between 0 and 1 (Example: 0.7) and a short explanation for the score.

Applicant Resume:
-----------------
{resume_text}
-----------------

Job Key Areas of Responsibility:
-----------------
1. Strategic Marketing Leadership
Proven CPG experience.
Ability to work with clients to develop strategic marketing plans.
Expertise in various marketing solutions (online, offline, mobile, social, etc.).
Research and insights interpretation.
Broad industry knowledge and client category expertise.
2. Building and Growing Client Relationships
Understanding clients' businesses.
Building confidence and credibility.
Achieving "trusted advisor" status.
Developing multi-tiered agency-client relationships.
3. Team Management
Managing and motivating marketing professionals.
Fostering teamwork and innovation.
Understanding project details for accountability.
4. Financial Performance Management
Achieving revenue targets.
Managing and planning accounts.
Working with business development teams for new opportunities.
5. Agency Contribution
Desire to see clients succeed.
Presentation skills.
Balancing client demands with agency initiatives.
Participating in agency leadership goals.
6. Experience Requirements
College degree (MBA preferred).
Approximately 7+ years in the industry (brand or direct marketing experience).
CPG and grocery retail experience.
Proven strategic skills and insights.
Strong knowledge of online and offline disciplines.
Interpersonal/communication skills.
Presentation development skills.
Ability to manage multiple projects and tight deadlines.
Openness to travel.
-----------------

Here is a sample "high-fit" resume with a score of 0.99 for reference:
-----------------
Sample High-Fit Resume for CPG Account Strategist at CEMM
[Full Name]

Email: [Email Address]
Phone: [Phone Number]
LinkedIn: [LinkedIn Profile]
Summary:
Highly experienced marketing and sales professional with over 7 years of proven success in CPG and grocery retail. 
Specialized in developing strategic marketing plans, building client relationships, and leading high-performing teams. 
Adept in online, offline, mobile, and social marketing strategies. Committed to delivering client success and achieving revenue targets.

Professional Experience:

Senior Sales Professional (From Alexandre Alves's Resume)

Experienced in Account and Sales Management within the CPG market.
Managed operations encompassing sales, logistics, and distribution.
Guided teams to achieve objectives within tight timelines.
Impact-driven Strategist (From Basem Ebied's Resume)

Specialized in driving sales and brand growth.
Collaborated with cross-functional teams to optimize user experiences.
Delivered successful campaign outcomes and management results.
Senior Marketing Manager (From John M. Robertson's Resume)

20+ years of solid marketing experience building strong brands.
Developed strategic and integrated brand plans.
Created effective brand positioning and executed profitable growth strategies.
Product Marketing Manager (From Kristin Leon's Resume)

Developed and executed marketing and portfolio strategy for CBG brands.
Managed commercialization and localization of product workflow.
Education:

Bachelor's Degree in Marketing: [University Name]
MBA (Preferred): [University Name]
Skills:

Strong interpersonal/communication skills.
Excellent presentation development and delivery skills.
Ability to manage multiple concurrent projects and meet tight deadlines.
Strong work ethic, integrity, and good business acumen.
Additional Information:

Open to travel as needed.
Strong knowledge of upper and lower funnel strategies and tactics.
Confidence without ego.
-----------------

Remember, your task is to determine a job fit score as a float between 0 and 1 (Example: 0.9) and a short explanation for score.
Respond with only the score and explanation. Do not include the resume or job description in your response.

Job Fit Score:
    """

    user_prompt = HumanMessagePromptTemplate.from_template(template=template)
    chat_prompt = ChatPromptTemplate.from_messages([user_prompt])
    formatted_prompt = chat_prompt.format_prompt(input=resume_text).to_messages()
    llm = llm
    result = llm(formatted_prompt)
    return result.content


def parse_score_and_explanation(result_content):
    # Assuming the score and explanation are on separate lines
    lines = result_content.split("\n")
    score = float(lines[0])  # Assuming the score is on the first line
    explanation = lines[1] if len(lines) > 1 else ""  # Explanation on the second line
    return score, explanation


def categorize_score(score, threshold1=0.8, threshold2=0.5):
    if score > threshold1:
        return "best"
    elif score > threshold2:
        return "maybe"
    else:
        return "low"


def save_to_category_folder(resumes_directory, resume, category, result_content):
    # Determine the applicant's name from the resume file name
    applicant_name = os.path.splitext(resume)[0]
    scores_directory = os.path.join(
        resumes_directory, "scores", category, applicant_name
    )
    os.makedirs(
        scores_directory, exist_ok=True
    )  # Ensures the folder is created if it doesn't exist
    original_pdf_path = os.path.join(resumes_directory, resume)
    new_pdf_path = os.path.join(scores_directory, resume)
    shutil.copy(original_pdf_path, new_pdf_path)
    text_file_path = os.path.join(scores_directory, f"{applicant_name}_response.txt")
    with open(text_file_path, "w", encoding="utf-8") as text_file:
        text_file.write(result_content)
    print(f"Saved PDF and response to {scores_directory}")


def main(resumes_directory, folder_path):
    # Ensure the main directory exists
    if not os.path.exists(resumes_directory):
        print(
            f"The directory {resumes_directory} does not exist. Please make sure the path is correct."
        )
        return

    for resume in folder_path:
        try:
            resume_path = os.path.join(resumes_directory, resume)
            print(f"Processing resume: {resume_path}")
            resume_text = ingest_pdf(resume_path)
            result_content = get_score(resume_text)
            score, _ = parse_score_and_explanation(result_content)
            category = categorize_score(score)
            save_to_category_folder(resumes_directory, resume, category, result_content)
        except Exception as e:
            print(f"An error occurred while processing resume {resume}: {e}")
            continue


if __name__ == "__main__":
    resumes_directory = "resumes"
    if not os.path.exists(resumes_directory):
        os.makedirs(resumes_directory)  # Create main directory if it doesn't exist
    folder_path = os.listdir(resumes_directory)
    main(resumes_directory, folder_path)
