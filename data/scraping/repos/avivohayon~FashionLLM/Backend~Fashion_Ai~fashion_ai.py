import os
import openai
from dotenv import load_dotenv, find_dotenv
from langchain.chains import LLMChain
from langchain.llms import OpenAI, OpenAIChat
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from Backend.Fashion_Ai.prompt_template import first_template, breakdown_example
from langchain.output_parsers import ResponseSchema, StructuredOutputParser
from data.DataClasses import AIJsonLikeData

# may want it to be a singleton
# so i first 'train' the ai with more calls plus this way i can make sure only
# one worker will use my api and not many for more control
class FashionAi:
    def __init__(self):
        load_dotenv(find_dotenv())
        openai.api_key = os.environ['OPENAI_API_KEY']

        # init the type of ai model
        self.__turbo_llm = ChatOpenAI(
                         temperature=0,
                         model_name='gpt-3.5-turbo'
                        )
        # build the structured dynamic prompt template for the ai
        self.fist_template = first_template
        self.breakdown_example = breakdown_example
        self.response_schemas = self._init_response_schemas()
        self.output_parser = StructuredOutputParser.from_response_schemas(self.response_schemas)
        self.format_instructions = self.output_parser.get_format_instructions()
        self.example_prompt_template = self._init_example_prompt_template()
        # feed the ai model with the structured dynamic prompt template
        # this will be our worker ai model
        self.fashion_llm = LLMChain(llm=self.__turbo_llm,
                                    prompt=self.example_prompt_template)

    def _init_response_schemas(self) -> list[ResponseSchema]:
        name_data_scheme = ResponseSchema(name='name',
                                          description="This is the name of the celebrity"
                                          )
        gender_data_scheme = ResponseSchema(name='gender',
                                            description="This is the gender of the celebrity, can be men, women or both")
        hat_data_scheme = ResponseSchema(name='hat',
                                         description="This is the data for the hat"
                                         )
        glasses_data_scheme = ResponseSchema(name='glasses',
                                             description="This is the data for the glasses"
                                             )
        jewelry_data_scheme = ResponseSchema(name='jewelry',
                                             description="This is the data for the jewelry"
                                             )
        tops_data_scheme = ResponseSchema(name='tops',
                                          description="This is the data for the tops"
                                          )

        pants_data_scheme = ResponseSchema(name='pants',
                                           description="This is the data for the pants"
                                           )

        shoes_data_scheme = ResponseSchema(name='shoes',
                                           description="This is the data for the shoes"
                                           )
        colors_data_scheme = ResponseSchema(name='colors',
                                            description="This is the data for the colors"
                                            )

        conclusion_data_scheme = ResponseSchema(name='conclusion',
                                                description="This is the data for the overall fashion breakdown in the end"
                                                )
        response_schemas = [
            name_data_scheme, gender_data_scheme, hat_data_scheme, glasses_data_scheme,
            jewelry_data_scheme, tops_data_scheme, pants_data_scheme,
            shoes_data_scheme, colors_data_scheme, conclusion_data_scheme
        ]
        return response_schemas

    def _init_example_prompt_template(self) -> PromptTemplate:
        example_prompt = PromptTemplate(
            template=self.fist_template,
            input_variables=["celebrity_name"],
            partial_variables={"format_instructions": self.format_instructions, "breakdown_example": self.breakdown_example},
            output_parser=self.output_parser
        )
        return example_prompt

    def get_llm_prediction(self, celebrity_name:str = 'Ozzy Osbourne') -> AIJsonLikeData:
        """

        :param celebrity_name:
        :return:
        """
        response = self.fashion_llm.predict_and_parse(celebrity_name= celebrity_name)
        return response
