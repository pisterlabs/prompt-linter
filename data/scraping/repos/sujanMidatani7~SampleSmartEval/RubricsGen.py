from guidance import Program
import os
import json

core_program: Program = Program(
    '''
        {{#system~}}
        You are an expert system in Evaluating the answer provided by an interviewee in an interview.
        Based on the question given,you can generate appropriate grading measures.
        You are very skilled in generating grading measures for the questions accurately and justifiably.
        {{~/system}}
        {{#user~}}
        Now, you are provided with Interviewee's Question.
        You are now asked to generate suitable/appropriate grading measures for the question and grade his answer according to them.
        The Question asked as follows:
        {{question}}
        Now, generate the grading measures.
        The grading measures must be generated as an array elements with names, the scale between 1-5, and weight of each rubrics in the range(0 to 1) as the grading rubrics. 
        They should be in """""json format""""".
        
        {{~/user}}
        {{#assistant~}}
        {{gen 'grading_measures' temperature=0.7 max_tokens=1500}}
        {{~/assistant}}
        ''', async_mode=True
)
