import guidance
from dotenv import load_dotenv

load_dotenv()

def indiQuesGrade(question, answer, role, exp):
    """
    Performs evaluation and grading of an interviewee's answer based on the provided question, answer, role, and experience.

    Args:
        question (str): The question asked to the interviewee.
        answer (str): The answer provided by the interviewee.
        role (str): The job role the interviewee applied to.
        exp (str): The years of experience the interviewee has in the job role.

    Returns:
        tuple: A tuple containing the evaluation results and the generated grading measures.
            - evaluation (dict): The evaluation of the answer in JSON format, containing grading measures, scores, and reasons.
            - grading_measures (list): The list of generated grading measures as array elements.
    """
    evaluatorModel = guidance.llms.OpenAI('gpt-3.5-turbo')
    evaluationSys = guidance('''
    {{#system~}}
    You are an expert system in evaluating the answer provided by an interviewee in an interview.
    Based on the question, answer given with information of applying role and years of experience, you can grade the answer on appropriate grading measures.
    You are very skilled in grading the answers accurately and justifiably.
    {{~/system}}
    {{#user~}}
    Now, you are provided with the interviewee's question, the job role they applied to, and their years of experience in it.
    You are now asked to generate suitable/appropriate grading measures for the question and grade their answer according to them.
    The question asked is as follows:
    {{question}}
    The role they applied to is as follows:
    {{role}}
    The years of experience they have in it is as follows:
    {{experience}}
    Now, generate the grading measures according to the above question, role, and experience values.
    The grading measures must be generated as array elements with names as the grading rubrics. They are placed between two square brackets, separated by commas.
    Do not output the grading measures yet.
    {{~/user}}
    {{#assistant~}}
    {{gen 'grading_measures' temperature=0.7 max_tokens=150}}
    {{~/assistant}}
    {{#user~}}
    Here's the answer provided by the interviewee in the interview:
    {{answer}}
    Now, perform the evaluation on the answer according to the generated grading measures.
    Output the evaluation in a JSON format with the grading measure as the key and a dictionary of score and reason as the value.
    The score key contains a numerical measure depicting the answer against the grading measure, and the reason key contains text information
    about why the answer was given such a numerical grade in the evaluation measure.
    Add the key of overall score to the output JSON with a dictionary as its value. The dictionary must have two keys: 'score', depicting the numerical measure
    as an overall evaluation score, graded against a score of 5, and 'reason', showing a justification statement.
    The output response must only contain a JSON file of the evaluation. Do not output any additional information other than it.
    {{~/user}}
    {{#assistant~}}
    {{gen 'evaluation' temperature=0.5 max_tokens=1500}}
    {{~/assistant}}
    ''', llm=evaluatorModel)

    output = evaluationSys(question=question, role=role, experience=exp, answer=answer)
    return output['evaluation'], output['grading_measures']
