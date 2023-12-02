from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from backend.resume_generation import ResumeGenerator
from backend import config


class FeedbackBot():
    def __init__(self, llm, generator):
        self.generator = generator
        self.verbose = False
        # get a revision model with memory
        with open(config.REVISE_PROMPT_TEMPLATE_PATH) as f:
            REVISE_PROMPT_TEMPLAT = f.read()
        revise_template = PromptTemplate(
            input_variables=["old_template", "user_request"],
            template=REVISE_PROMPT_TEMPLAT,
        )
        feedback_memory = ConversationBufferMemory(
            input_key='user_request', memory_key='query_history')

        self.revision_chain = LLMChain(llm=llm,
                                       memory=feedback_memory,
                                       prompt=revise_template,
                                       verbose=self.verbose)

    def update(self, path):
        """
        revise the template with protected keywords
        """
        # protect keywords
        with open(path) as f:
            template = f.read().replace('{', '#(#').replace('}', '#)#')
        # revise the template
        template = self.revision_chain.run(
            old_template=template, user_request=self.user_query)
        # protect keywords        
        with open(path, 'w') as f:
            f.write(template.replace('#(#', '{').replace('#)#', '}'))

    def run(self, user_query, verbose=False):
        """
        update all prompts
        """
        self.verbose = verbose
        self.user_query = user_query

        self.update(config.REPHRASE_PROMPT_TEMPLATE_PATH)
        self.update(config.LATEX_PROMPT_TEMPLATE_1_PATH)
        self.update(config.LATEX_PROMPT_TEMPLATE_2_PATH)
        
        self.update(config.JOB_SUMMARY_PROMPT_PATH)
        self.update(config.BACKGROUND_SECTION_TEMPLATE_1_PATH)
        self.update(config.BACKGROUND_SECTION_TEMPLATE_2_PATH)

        self.generator.run('')
