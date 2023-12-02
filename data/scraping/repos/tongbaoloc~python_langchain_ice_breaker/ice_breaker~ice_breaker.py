import os

from langchain import PromptTemplate, LLMChain
from langchain.chat_models import ChatOpenAI

from agents import linkedin_lookup_agent
from third_parties.linkedin import get_linkedin_data

# information = """
#     Elon Reeve Musk (/ˈiːlɒn/ EE-lon; born June 28, 1971) is a business magnate and investor. He is the founder, chairman, CEO and chief technology officer of SpaceX; the angel investor, CEO and product architect of Tesla, Inc.; the owner and CTO of Twitter; the founder of the Boring Company; the co-founder of Neuralink and OpenAI; and the president of the Musk Foundation. Musk is the wealthiest person in the world, with an estimated net worth of US$239 billion as of July 2023, according to the Bloomberg Billionaires Index, and $248.8 billion according to Forbes's Real Time Billionaires list, primarily from his ownership stakes in Tesla and SpaceX.[4][5][6]
#
# Musk was born in Pretoria, South Africa, and briefly attended the University of Pretoria before moving to Canada aged 18, acquiring citizenship through his Canadian-born mother.[citation needed] Two years later, he matriculated at Queen's University in Kingston, Ontario, and two years after that transferred to the University of Pennsylvania, where he received bachelor's degrees in economics and physics. He moved to California in 1995 to attend Stanford University. After two days, he dropped out and, with his brother Kimbal, co-founded the online city guide software company Zip2. The startup was acquired by Compaq for $307 million in 1999, and with $12 million of the money he made, that same year Musk co-founded X.com, a direct bank. X.com merged with Confinity in 2000 to form PayPal.
#
# In 2002, eBay acquired PayPal for $1.5 billion, and that same year, with $100 million of the money he made, Musk founded SpaceX, a spaceflight services company. In 2004, he was an early investor in the electric vehicle manufacturer Tesla Motors, Inc. (now Tesla, Inc.). He became its chairman and product architect, assuming the position of CEO in 2008. In 2006, he helped create SolarCity, a solar energy company that was acquired by Tesla in 2016 and became Tesla Energy. In 2013, Musk proposed a hyperloop high-speed vactrain transportation system. In 2015, he co-founded OpenAI, a nonprofit artificial intelligence research company. The following year, he co-founded Neuralink—a neurotechnology company developing brain–computer interfaces—and the Boring Company, a tunnel construction company. In 2022, his acquisition of Twitter for $44 billion was completed. In 2023, Musk founded xAI, an artificial intelligence company.
#
# Musk has expressed views that have made him a polarizing figure. He has been criticized for making unscientific and misleading statements, including that of spreading COVID-19 misinformation, and promoting conspiracy theories[7][8][9] and transphobic content.[10] In 2018, the U.S. Securities and Exchange Commission (SEC) sued Musk for falsely tweeting that he had secured funding for a private takeover of Tesla. To settle the case, Musk stepped down as the chairman of Tesla and paid a $20 million fine.
# """

if __name__ == "__main__":
    summary_template = """"
        given the Linkedin information {information} about a person from I want you to create:
        1. a short summary of the person
        2. why the person is famous?
    """

    summary_prompt_template = PromptTemplate(
        input_variables=["information"], template=summary_template)

    llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo", openai_api_key= os.getenv("OPENAI_API_KEY"))

    chain = LLMChain(llm=llm, prompt=summary_prompt_template)

    # Tu lay  linkined url tu ho va ten
    linkedin_profile_url = linkedin_lookup_agent.lookup("Learning Vault")
    # linkedin_profile_url = "https://linkedin.com/in/bao-loc-tong-a523372a/"
    data = get_linkedin_data(linkedin_profile_url)
    # print(data)
    print(chain.run(information=data))
