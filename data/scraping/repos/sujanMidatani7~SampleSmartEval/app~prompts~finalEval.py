from guidance import Program

core_program: Program = Program(
    """
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
        Here are the list of records made from questions answered with grades under appropriate rubrics. These records also
        contain wieght of eaxh rubrics and scale.
        Finally, the records contain a float value of the plagiarism score. We have set the threshold of 0.96 for an answer to be considered plagiarized.
        
        The List of evaluations records are as follows ::
        {{ires}}
        {{~/user}}
        {{#user~}}
        Based on the above inputs of the interview, generate an overall performance score and scope of improvements based on it.which provides the accurate result and analysis of performence.
        {{~/user}}
        {{#assistant~}}
        {{gen 'final_evaluation' temperature=0.5 max_tokens=1000}}
        {{~/assistant}}
    """,
    async_mode=True,
)
