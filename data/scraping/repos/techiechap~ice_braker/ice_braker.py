from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI

from langchain.chains import LLMChain
from agents.linkedin_lookup_agent import lookup as linkedin_lookup_agent
from third_parties.linkedin import scrape_linkedin_profile



if __name__ == "__main__":
    print("Hello Langchain")
# print(os.environ['OPENAI_API_KEY'])

linkedin_profile_url = linkedin_lookup_agent(name="Edan Marco")

summary_template = """
    given the linkedin information {information} about a person , create a 
    1. a short summary
    2. 2 intresting fact
    3. Name of Organization person belogs to
"""
summary_prompt_template = PromptTemplate(
    input_variables=["information"], template=summary_template
)

llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")

chain = LLMChain(llm=llm, prompt=summary_prompt_template)


# print(chain.run(information=information))


linkedin_data = scrape_linkedin_profile(
    linkedin_profile_url=linkedin_profile_url                     #"https://www.linkedin.com/in/elon-musk-69a268206/"
)

print(chain.run(information=linkedin_data))

#
# print(linkedin_data) uncomment this to see 200 response
#
# print(linkedin_data.json()) # This will get the entire response
