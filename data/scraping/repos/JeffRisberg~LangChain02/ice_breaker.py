from langchain import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain

from third_parties.linkedin import scrape_linkedin_profile


if __name__ == "__main__":
    summary_template = """
      Given the LinkedIn information {information} about a person, I want to you create:
      1. a short summary of the person
      2. two interesting facts about the person

    """

    summary_prompt_template = PromptTemplate(
        input_variables = ["information"], template = summary_template)

    llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")

    chain = LLMChain(llm=llm, prompt=summary_prompt_template)

    linkedin_information = scrape_linkedin_profile(linkedin_profile_url="https://www.linkedin.com/in/elonmusk/")

    print(chain.run(information=linkedin_information))



