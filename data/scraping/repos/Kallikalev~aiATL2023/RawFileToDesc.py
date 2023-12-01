from google.cloud import aiplatform
from langchain.llms import VertexAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

def rawToDesc(skills, repoName, readMe, fileName, rawFile): #skills string, rawFile string, repoDesc string
    template = """
        You are an expert technical recruiter screening system. Your job is to evaluate a file from the github of a candidate and determine their strengths and weaknesses based on this file.

        Below is a list of skills listed by an applicant on their resume, the name of one of their github repositories, the repository ReadMe (if it exists), and the full text of a file they wrote.

        Summarize the file and how it relates to the repository as a whole. Evaluate the skills which are displayed in this file with a brief analysis of strengths and weaknesses, as well as a score of 1 to 10. If a skill is not shown in this file, do not list it.

        Skills: {skills}

        Repository name: {repoName}

        ReadMe: {readMe}

        File name: {fileName}

        File Contents:

        {rawFile}

        Provide your response below, in the following format (Remember to be insightful, unique, and consise in your analysis, and if a skill is not shown in this file, DO NOT LIST IT). Limit to a maximum of 6 skills:

        File summary: <summary>

        Skill 1: <skill analysis>

        Skill 2: <skill analysis>

        Skill 3: <skill analysis>

        Skill 4: <skill analysis>

        ...

    """

    llm = VertexAI(model_name="code-bison",max_output_tokens=1000,temperature=0.3)
    prompt = PromptTemplate(template=template, input_variables=["skills","repoName","readMe","fileName","rawFile"])


    llm_chain = LLMChain(prompt=prompt,llm=llm)
    if readMe is None:
        readMe = "No ReadMe found"
    response = llm_chain.run({"skills":skills,"repoName":repoName,"readMe":readMe,"fileName":fileName,"rawFile":rawFile})
    return response


