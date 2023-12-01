from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI
from utils.json_utils import llm_response_schema
from apikey import load_env
OPENAI_API_KEY, SERPER_API_KEY = load_env()


class AutoMarketAnalysisPromptGenerator:
    def __init__(
            self, 
            llm : OpenAI | ChatOpenAI | None = None, 
            data_path : str = 'data/integrated/market_analysis'
    ):
        embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
        chroma_instance = Chroma(persist_directory=data_path, embedding_function=embeddings)
        if llm is None:
            llm = OpenAI(model_name="gpt-3.5-turbo", temperature=0, openai_api_key=OPENAI_API_KEY)

        self.qa_model = RetrievalQA.from_chain_type(
            llm = llm,
            chain_type="stuff",
            retriever=chroma_instance.as_retriever()
        )

        self.prompt_template = '''
            Craft a paragraph of how chatgpt (address as you) supposed to act based on the role stated. 
            Comprehensive market research and analysis to understand the {0} industry and identify potential opportunities and challenges.
            provides detail questions.
            The paragraph should contain "I want you to act as a "
        '''

    def generate_dynamic_prompt(
            self, 
            domain : str
    ) -> str:
        knowledge = "senior in {}".format(domain)
        prompt = self.qa_model.run(self.prompt_template.format(domain, knowledge))
        prompt = (
            f'{prompt}\n' + 
            f'Constraints: \n' +
            f'Maximum 400 words for response, try to summarize as much as possible\n' +
            f'Performance Evaluation: \n' +
            f'Every command has a cost, so be smart and efficient. Aim to complete tasks in the least number of steps. Minimum search step.\n'
            
        )
        return prompt


class AutoCompetitorAssessmentPromptGenerator:
    def __init__(
            self, llm : ChatOpenAI | None = None, 
            data_path : str = 'data/integrated/competitor_assessments'
    ):
        embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
        chroma_instance = Chroma(persist_directory=data_path, embedding_function=embeddings)
        if llm is None:
            llm = OpenAI(model_name="gpt-3.5-turbo", temperature=0, openai_api_key=OPENAI_API_KEY)

        self.qa_model = RetrievalQA.from_chain_type(
            llm = llm,
            chain_type="stuff",
            retriever=chroma_instance.as_retriever()
        )

        self.prompt_template = '''
            Craft a paragraph that
            identify keyword opportunities by analyzing the top-ranking keywords for each competitor on {0}.
            Privides detail question.
        '''

    def generate_dynamic_prompt(
            self, 
            domain : str, 
    ) -> str:
        prompt = self.qa_model.run(self.prompt_template.format(domain))
        prompt = (
            f'{prompt}\n' + 
            f'Search on google some competitor on {domain} and answer:' + 
            f'1. Who/which is Competitors' +
            f'2. Strategies of theme' + 
            f'3. Ads Strategies' + 
            f'4. Differentiators and Customer Experience' +
            f'Constraints: \n' +
            f'Maximum 400 words for response, try to summarize as much as possible\n' +
            f'Performance Evaluation: \n' +
            f'Every command has a cost, so be smart and efficient. Aim to complete tasks in the least number of steps. Minimum search step.\n'  
        )
        return prompt
    
class AutoDetectUniqueSellingPointPromptGenerator:
    def __init__(
            self, llm : ChatOpenAI | None = None, 
            data_path : str = 'data/integrated/unique_selling_point'
    ):
        embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
        chroma_instance = Chroma(persist_directory=data_path, embedding_function=embeddings)
        if llm is None:
            llm = OpenAI(model_name="gpt-3.5-turbo", temperature=0, openai_api_key=OPENAI_API_KEY)

        self.qa_model = RetrievalQA.from_chain_type(
            llm = llm,
            chain_type="stuff",
            retriever=chroma_instance.as_retriever()
        )

        self.prompt_template = '''
            With my competitor analysis {0}.
            Craft paragraph to ask chatgpt to analize the unique selling point of my {1} product.
        '''

    def generate_dynamic_prompt(
            self, 
            domain : str, 
            competitor_analysis: str
    ) -> str:
        prompt = self.qa_model.run(self.prompt_template.format(competitor_analysis, domain))
        prompt = (
            f'Search on internet and answer the questions:\n'
            f'{prompt}\n' + 
            f'Constraints: \n' +
            f'Maximum 400 words for response, try to summarize as much as possible\n' +
            f'Performance Evaluation: \n' +
            f'Every command has a cost, so be smart and efficient. Aim to complete tasks in the least number of steps. Minimum search step.\n'
            
        )
        return prompt
    

class AutoContentCreationPromptGenerator:
    def __init__(
            self, 
            llm : ChatOpenAI | None = None, 
            data_path : str = 'data/integrated/content_creator'
    ):
        embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
        chroma_instance = Chroma(persist_directory=data_path, embedding_function=embeddings)
        if llm is None:
            llm = OpenAI(model_name="gpt-3.5-turbo", temperature=0, openai_api_key=OPENAI_API_KEY)

        self.qa_model = RetrievalQA.from_chain_type(
            llm = llm,
            chain_type="stuff",
            retriever=chroma_instance.as_retriever()
        )

        self.prompt_template = '''
            Craft a paragraph of how chatgpt (address as you) supposed to act based on the role stated. 
            Provide expectation of the required scope, skillset and knowledge. 
            If there is no specific role found, use relative reference if necessary. 
            The role is craete a post in social media like {0}, the post must be most attracted to reader. \n\n
            Task: generate chatGPT prompt
            Goal: help chatGPT creat a content marketing in social media including some specific in this media such as #hashtag, websitelink, etc. \n
            The paragraph should include: "Write a post in {1} "
        '''

        self.llm_schema = llm_response_schema()

    def generate_dynamic_prompt(
            self, media : str, 
    ) -> str:
        prompt = self.qa_model.run(self.prompt_template.format(media, media))
        prompt = (
            f'{prompt}\n' + 
            f'Constraints: \n' +
            f'Maximum 400 words for response, try to summarize as much as possible\n' +
            f'If need to search on internet, limit to 3 times\n' +
            f'Performance Evaluation: \n' +
            f'Every command has a cost, so be smart and efficient. Aim to complete tasks in the least number of steps. Minimum search step.\n'
            f'Respond with only valid JSON conforming to the following schema:\n{self.llm_schema}\n'
        )
        return prompt
    