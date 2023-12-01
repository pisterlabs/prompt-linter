import guidance

def indiQuesGrade(question, answer, role, exp):
    # Initialize the evaluator model
    evaluatorModel = guidance.llms.OpenAI('gpt-3.5-turbo')
    
    # Define the evaluation system using the guidance template
    evaluationSys = guidance('''
    {{#system~}}
    You are an expert system in Evaluating the answer provided by an interviewee in an interview.
    Based on the question, answer given with information of Applying Role and Years of Experience, you can grade the answer on appropriate grading measures.
    You are very skilled in grading the answers accurately and justifiably.
    {{~/system}}
    {{#user~}}
    Now, you are provided with Interviewee's Question, his job role he applied to, and his years of experience he has with it.
    You are now asked to generate suitable/appropriate grading measures for the question and grade his answer according to them.
    The Question asked as follows:
    {{question}}
    The Role he applied to is as follows :
    {{role}}
    The years of experience he has in it is as follows :
    {{experience}}
    Now, generate the grading measures according to the above question, role and experience values.
    The grading measures must be generated as a array elements with names as the grading rubrics. They are placed between two square brackets, separated by commas.
    Do not output the grading measures yet.
    {{~/user}}
    {{#assistant~}}
    {{gen 'grading_measures' temperature=0.7 max_tokens=150}}
    {{~/assistant}}
    {{#user~}}
    Here's the answer provided by the interviewee in the interview :
    {{answer}}
    Now, perform the evaluation on the answer according to the generated grading measures.
    Output the evaluation in a JSON Format with the grading measure as key and a dictionary of score and reason as value.
    The score key contains a numerical measure depicting the answer against grading measure and the reason key contains text information
    about why the answer was such given such numerical grade in the evaluation measure.
    Add the key of overall score to the output JSON with a dictionary as it's value. The dictionary must have two keys, score, depicting the numerical measure
    as a overall evaluations score, graded against a score of 5 and the other key as reason, showing a Justification Statement.
    The output response must only contain a JSON File of evaluation. Do not output any additional information other than it.
    {{~/user}}
    {{#assistant~}}
    {{gen 'evaluation' temperature=0.5 max_tokens=1500}}
    {{~/assistant}}
    ''', llm = evaluatorModel)
    
    # Call the evaluation system with the provided inputs
    output = evaluationSys(question=question, role=role, experience=exp, answer=answer)
    
    # Return the evaluation and grading measures
    return output['evaluation'], output['grading_measures']
