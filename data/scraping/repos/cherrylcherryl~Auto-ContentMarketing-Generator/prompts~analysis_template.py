
from langchain import PromptTemplate
from utils.prompt_utils import add_market_analysis_constraint, add_competitor_analysis_constraint, add_key_selling_point_analysis_constraint
class StaticPromptTemplate:
    def __init__(self):
        self.MARKET_ANALYSIS_PROMPT = PromptTemplate(
            input_variables=["company", "domain"],
            template='''
            My company is {company} and working on {domain}, in this market what is my chance and challenge
            '''
        )

        self.COMPETITORS_PROMPT = PromptTemplate(
            input_variables=["company"],
            template='''What are top 5 competiors of {company} company.
            '''
        )

        self.KEY_SELLING_POINT = PromptTemplate(
            input_variables=["company"],
            template='''
            Act as a senior, you must be thinking carefully to answer this question.
            Your resource is not limited such as internet and google search.
            The question is: "What are the key selling points of {company}'s products? Try to generalize and not base it off one product."
            '''
        )
    
    def get_market_analysis_prompt(
            self,
            company: str,
            domain : str
        ) -> str:
        prompt = self.MARKET_ANALYSIS_PROMPT.format(company=company, domain=domain)
        prompt = add_market_analysis_constraint(
            base_prompt=prompt
        )
        return prompt
    
    def get_competitor_prompt(
            self, 
            company : str
        ) -> str:
        prompt = self.COMPETITORS_PROMPT.format(company=company)
        # prompt = add_competitor_analysis_constraint(
        #     base_prompt=prompt
        # )
        return prompt
    
    def get_key_selling_point(
            self, 
            company : str
        ) -> str:
        prompt = self.KEY_SELLING_POINT.format(company=company)
        prompt = add_key_selling_point_analysis_constraint(
            base_prompt=prompt
        )
        return prompt

