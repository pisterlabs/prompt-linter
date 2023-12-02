import guidance
import pandas as pd
from dotenv import load_dotenv

load_dotenv() 
def finalGradingPrompt(resume_summary, role, exp, ires):
    """
    Evaluates the interview scenario and provides an overall score and scope of improvement statements for the interviewee.

    Args:
        resume_summary (str): The extracted JSON summary of the interviewee.
        role (str): The role the interviewee applied for.
        exp (str): The experience of the interviewee in the applied role.
        ires (list): A list of CSV records containing questions, grades, timestamps, and plagiarism scores.

    Returns:
        dict: A dictionary containing the final evaluation score and scope of improvement statements.
    """
    # Initialize the guidance model
    model = guidance.llms.OpenAI('gpt-3.5-turbo')
    
    # Define the final grading prompt using the guidance template
    finalRes = guidance('''
        {{#system~}}
        You are now the Final Decision Interview Result Grading Expert. You are provided with an Interview's evaluation details.
        You need to evaluate the interview scenario and provide an overall score and set of Scope of Improvement statements for the interviewee.
        {{~/system}}
        {{#user~}}
        The interview has been completed and the results of the interview will be provided to you. You need to evaluate the case and 
        provide an overall score of the interviewee's performance and suggestions for further improvements if required, based on the overall score.
        Here's the Interviewee's Extracted JSON Summary:
        {{resume_summary}}
        {{~/user}}
        {{#user~}}
        The interviewee applied to the following role:
        
        {{role}}
        and has the following experience in that role:
        {{exp}}
        Here are the list of CSV records made from questions answered with grades under appropriate rubrics. These records also
        contain the start and end timestamps of the interviewee answering the questions within a 2-minute time constraint.
        Finally, the records contain a float value of the plagiarism score. We have set the threshold of 0.96 for an answer to be considered plagiarized.
        
        The List of evaluations records are as follows ::
        {{ires}}
        {{~/user}}
        {{#user~}}
        Based on the above inputs of the interview, generate an overall performance score and scope of improvements based on it.
        {{~/user}}
        {{#assistant~}}
        {{gen 'final_evaluation' temperature=0.5 max_tokens=1000}}
        {{~/assistant}}
    ''', llm=model)
    
    # Calling the final grading prompt with the provided inputs
    res = finalRes(resume_summary=resume_summary, role=role, exp=exp, ires=ires)
    
    # Return the final evaluation from the response
    return res['final_evaluation']
def get_shape(csv_file, resume_summary, role, experience):
    """
    Reads a CSV file and performs final grading prompt based on the provided inputs.

    Args:
        csv_file (file): The CSV file containing evaluation records.
        resume_summary (str): The extracted JSON summary of the interviewee.
        role (str): The role the interviewee applied for.
        experience (str): The experience of the interviewee in the applied role.

    Returns:
        dict: A dictionary containing the final evaluation score and scope of improvement statements.
    """
    with open(csv_file.name, "r") as f:
        k = pd.read_csv(f)
        k.dropna()
        l = finalGradingPrompt(resume_summary=resume_summary, role=role, exp=experience, ires=k.values.tolist())
        return l
