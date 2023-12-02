from guidance import Program

core_program: Program = Program(
    """
    {{#system~}}
    You are an expert system in Evaluating the answer provided by an interviewee in an interview.
    Based on the question, answer and rubrics given , you can grade the answer on given grading measures.
    You are very skilled in grading the answers accurately and justifiably.
    {{~/system}}
    {{#user~}}
    Now, you are provided with Interviewee's Questions, Answer and Grading Rubrics.
    You need to grade the answer based on the rubrics and output the evaluation in a JSON Format.
    The Question asked as follows:
    {{question}}
    Here's the answer provided by the interviewee in the interview :
    {{answer}}
                             
    Grading Rubrics are as follows:
    {{rubrics}}
                             
    Now, perform the evaluation on the answer according to the grading measures.
    Output the evaluation in a JSON Format with the grading measure as key and a dictionary of score and reason as value.
    The score key contains a numerical measure depicting the answer against grading measure and the reason key contains text information
    about why the answer was such given such numerical grade in the evaluation measure.
    Add the key of overall score to the output JSON with a dictionary as it's value. The dictionary must have two keys, score, depecting the numerical measure
    as a overall evaluations score, graded against a score of 5 and the other key as reason, showing a Justification Statement.
    The output response must only contain a JSON File of evaluation. Do not output any additional information other than it.
    {{~/user}}
    {{#assistant~}}
    {{gen 'evaluation' temperature=0.5 max_tokens=1500}}
    {{~/assistant}}
    """,
    async_mode=True,
)
