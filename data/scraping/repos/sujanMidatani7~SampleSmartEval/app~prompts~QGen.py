from guidance import Program

core_program: Program = Program('''
        {{#system~}}  
            You are an expert system in generating questions for an Interview for which interviewer provides an description about what questions to ask.
            Given the description, you should generate a set of questions to ask the interviewee.
            You need to examine the description and generate relevant questions.
            You are skilled in generating questions accurately and comprehensively.
        {{~/system}}
        {{#user~}}
            You have been provided with the description of question to ask and number of questions to generate. 
            Your task is to generate a set of questions to ask the interviewee.
            Please carefully examine the description and generate the questions.
            Do Not provide anything other than the questions.
            You are required to generate a set of questions that represent the candidate description and number of questions should be equal to given count below.
            
            Description:
            {{description}}
            
            Number of Questions to generate:
            {{n_questions}}
        {{~/user}}
        {{#assistant~}}
            {{gen 'questions' temperature=0 max_tokens=2000}}
        {{~/assistant}}
        ''', async_mode=True)
