from langchain import PromptTemplate, LLMChain
from langchain.chat_models import ChatOpenAI

def analyze_desired_candidate_profile(job_description, temperature=0.1, chosen_model="gpt-4"):
    llm = ChatOpenAI(temperature=temperature, model_name=chosen_model)

    template = PromptTemplate(
        input_variables=["job_description"],
        template="""
        Your objective is to extract key technical stacks, relevant job details, and pertinent company information in YAML format from the provided job description. Please adhere to the following guidelines:

        1. The names of technical stacks, tools, and frameworks (e.g., Java, Rust, Postgres) are of utmost importance. Ensure to include these keywords in the output.
        2. Keywords related to job functions are critical. Specifically, extract terms like "backend", "frontend", "machine learning", "UI design", and so on.
        3. Exclude keywords related to benefits for the job seeker, such as compensation, benefits, leave policies, etc.
        4. Extract and highlight any mentions of the company name or other relevant information that a candidate should remember when applying.
        5. Lastly, extract and present the desired_candidate_profile based on the current prompt's findings.

        Based on the input below, output the information adhering to the above guidelines in YAML format.
        
        Please provide the structured data in YAML format without including the "```yaml" notation.


        ---

        {job_description}

        ---
    """
    )

    chain = LLMChain(llm=llm, prompt=template, verbose=True)
    result = chain.predict(job_description=job_description)

    return result


def refine_resume(desired_candidate_profile, resume, temperature=0.1, chosen_model="gpt-4"):
    llm = ChatOpenAI(temperature=temperature, model_name=chosen_model)

    template = PromptTemplate(
        input_variables=["desired_candidate_profile", "resume"],
        template="""
        Given the specific "Desired Candidate Profile" provided below, I need you to rewrite and enhance the content of the provided "Resume" to make it more appealing.
        
        ## Resume
        {resume}        
        ---
        ## Desired Candidate Profile
        {desired_candidate_profile}
        ---

        Recruiters and hiring managers typically don't go through every application document in detail. Instead, they have software tools that use specific keywords to compare job listings with resumes and display the ones that match the most.

        Your task is to employ the same approach using keywords and highlight the experiences the hiring entity values the most, tailoring the resume to each specific job listing.

        If the resume contains qualifications that match the job listing, emphasize them attractively. For instance, if the job listing emphasizes SQL experience, ensure the resume aligns with their post.

        Even if it's not a perfect match, if there's something close enough, craft sentences that incorporate those keywords without being deceptive.

        Present the qualifications in a manner that aligns exceptionally well with the requirements of the specific job.

        The output should be in a structured text format using Markdown. For instance, under "Work Experience", list the job history chronologically. The same goes for "Education".

        If the original resume contains the applicant's name, contact details, and links like LinkedIn, ensure these are displayed above other items.

        If the resume includes critical prerequisites like visa status, these should be mentioned near the top.

        Keep the resume content concise, fitting it to the equivalent of one A4 page.        
        
        Please provide the structured data in Markdown format without including code blocks or other unnecessary notations.


    """
    )

    chain = LLMChain(llm=llm, prompt=template, verbose=True)
    result = chain.predict(desired_candidate_profile=desired_candidate_profile, resume=resume)
    return result

