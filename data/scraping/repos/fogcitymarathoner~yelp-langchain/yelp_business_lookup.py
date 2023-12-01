import os
from langchain import PromptTemplate
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI

from third_parties.yelp import scrape_yelp_profile
from tools.tools import get_yelp_business_id


if __name__ == "__main__":
    yelp_business_id = get_yelp_business_id(
        "starbucks", "500 linda mar blvd", "pacifica", "ca", "US"
    )
    business_data = scrape_yelp_profile(yelp_business_id)

    summary_template = """
         given the {business_data} about a business from I want you to create:
         1. an is_closed status report by name only no title
     """

    summary_prompt_template = PromptTemplate(
        input_variables=["business_data"],
        template=summary_template,
    )
    
    llm = ChatOpenAI(temperature=1, model_name=os.environ.get('MODEL_NAME'))

    chain = LLMChain(llm=llm, prompt=summary_prompt_template)

    print(chain.run(business_data=business_data))
