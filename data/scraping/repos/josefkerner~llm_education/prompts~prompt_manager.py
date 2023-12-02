import re
import anthropic
from rec_service.data_model.cv import CV
from rec_service.data_model.role import Role
from rec_service.config.message_generator_cfg import MESSAGE_GENERATION_CFG, REC_GENERATION_CFG
from rec_service.generation_service.prompts.inContextExampleManager import inContextExampleManager
class PromptManager:

    def __init__(self):
        self.inContextManager = inContextExampleManager()
    def get_message_prompt(self, role: Role, cv: CV, include_examples=True):
        '''
        Obtains prompt
        :param role:
        :param cv:
        :return:
        '''
        assert cv.summary is not None
        assert role.summary is not None
        if MESSAGE_GENERATION_CFG['example_strategy'] == "langchain":
            prompt = self.inContextManager.select_examples(
                selection_strategy=MESSAGE_GENERATION_CFG['selection_strategy'],
                input_variables=["name","role_description", "cv_summary", "message"],
                prefix="""
                    Generate a message which will interest candidate into a role.
                    Write as if you are a recruiter.
                    Do not mention candidate education and certifications in the generated message.
                    If a candidate is not a fit for the job role, return string : "Not a fit".
                """,
                suffix="""
                    Candidate name: {name} Candidate experience: {cv_summary} Job role description: {role_description} Generated message: {message}
                    """,
                type='message',
                num_examples=MESSAGE_GENERATION_CFG['num_examples']
            )
            prompt = prompt.format(name=cv.name, cv_summary=cv.summary,role_description=role.summary,message="")

            prompt = prompt.replace('\n','').replace('\t','')
            prompt = re.sub(r' {2,}',' ',prompt)
            print(prompt)
            return prompt




    def get_rec_prompt(self, role: Role, cv: CV):
        '''
        Obtains prompt
        :param role:
        :param cv:
        :return:
        '''

        if REC_GENERATION_CFG['example_strategy'] == 'langchain':
            prompt = self.inContextManager.select_examples(
                input_variables=["name","surname","role_description", "cv_summary","rec"],
                selection_strategy=REC_GENERATION_CFG['selection_strategy'],
                prefix=""""
                Explain why is candidate a fit for the job role based on role requirements and candidate experience.
                If a candidate is not a fit for the job role, return string : "Not a fit". 
                """,
                suffix="""
                Candidate full name: {name}{surname}\n Candidate experience: {cv_summary}\n Job role description: {role_description}\n Generated recommendation: {rec}
                """,
                type='rec',
                num_examples=REC_GENERATION_CFG['num_examples']

            )

            prompt = prompt.format(
                name=cv.name,
                surname=cv.surname,
                cv_summary=cv.summary,
                role_description=role.summary,
                rec=""
            )
            return prompt

        if REC_GENERATION_CFG['model_type'] == 'gptturbo':
            prompt = [
                {
                    'role': 'system',
                    'content': 'You are helpfull bot which helps recruiters to find the best candidates for the given job.'
                },
                {
                    'role': 'assistant',
                    'content': f'Remember this job role requirements: {role.summary}'
                },
                {
                    'role': 'assistant',
                    'content': f'Remember this candidate cv: {cv.summary}'
                },
                {
                    'role': 'user',
                    'content': 'Explain why is candidate a fit for the job role based on role requirements and candidate experience.'
                }
            ]
        else:

            prompt = f"""
            Explain why is candidate a fit for the job role based on role requirements and candidate experience.
            If candidate is not a fit, explain why.
            Respond only with short sentences.
            Candidate name: {cv.name} {cv.surname},
            Candidate experience: {cv.summary}
            Job role requirements: {role.summary}
            Recommendation:
            """
            prompt = prompt.replace('\n', '').replace('\t', '').replace('  ', '')
        return prompt


    def get_question_prompt(self, role: Role, cv: CV):
        prompt_template = f"""
            Generate 10 questions which will verify if candidate has the required skills for the job role based on provided job role description.
            Generate deep technical questions which will verify candidate skills. Return questions only.
            Example:
            ------------
            Job role description: "We are looking for Data scientist with 3+ years of experience in Python, Machine Learning, SQL"
            Questions: 1. What is the difference between precision and recall? 2. What is a for comprehension in Python? 3. How Decision trees work? 4. Which NLP libraries do you know? 5. What are the steps in any data science project? 6. What can be wrong if model is reaching 99% accuracy on the negative class?
            ------------
            Job role description: {role.summary},
            Questions:
            

        """
        prompt = prompt_template.replace('\n', '').replace('\t', '').replace('  ', '')
        return prompt
    @staticmethod
    def get_cv_prompt(cv: CV):
        prompt_template= f"""
                \n\n{anthropic.HUMAN_PROMPT}Summarize a given CV and List candidate top skills (including years of practice), experience and education. Return only summary and only in English language. Include candidate name
                Candidate name: {cv.name} {cv.surname},
                CV: {cv.experience}
                Summary:{anthropic.AI_PROMPT}

        """
        prompt_template = prompt_template.strip().replace('\n', '').replace('\t', '').replace('  ', '')
        return prompt_template
