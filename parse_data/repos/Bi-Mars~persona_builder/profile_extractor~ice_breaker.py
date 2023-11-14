from langchain import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from dotenv import (
    load_dotenv,
    find_dotenv,
)  # These imports are used to find .env file and load them

from third_parties.linkedin import scrape_linkedin_profile  # we created this
from third_parties.twitter import scrape_user_tweets
from agents.linkedin_lookup_agents import lookup as linkedin_lookup_agent
from agents.twitter_lookup_agent import lookup as twitter_lookup_agent
from output_parser import person_intel_parser, PersonIntel


""" Step-0:API KEY via .env file
    1. Must install: pipenv install python-dotenv
    2. Find the .env file and load them onto your OS
"""
load_dotenv(find_dotenv())


def ice_break(name: str) -> tuple[PersonIntel, str]:
    """Step-1: Model
    1. Create an instance of a model. For this usecase we are using chatmodel.
    2. ChatModel is a wrapper around LLM
    3. Provide the Model_name  and temperature.
        - Temperature determines how creative the LLM will be. Sort of like defining a randomness
        - Model_name: What LLM model are you using?
            -- You can find the name of the model at [openAI Language Model](https://platform.openai.com/docs/models)

    """
    llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")

    """ Step-2: Prompt
        1. Engineer the prompt.
        2. Make the prompt dynamic: {dynamic_text} --> parameter
    """
    ######################################################################
    # and twitter {twitter_information}
    profile_summary_template = """
        Given the LinkedIn information {linkedin_information}  about a person, I want you to create:
            1. A short summary
            2. Two interesting facts about them
            3. Topics of interests
            4. 2 creative and personal ice breakers to open a conversation with them.
            \n{format_instructions}
    """

    ######################################################################
    """ Step-3: Prompt template
        1. Create a prompt template. 
            - Using Langchain's PromptTemplate: 
                -- It is a Schema to represent a prompt for an LLM.
        2. Provide a list of the names of the variables the prompt template expects.
            - input_variables
            
        3. partial_variable = Formatting information
    
    """
    profile_summary_prompt_template = PromptTemplate(
        input_variables=["linkedin_information"],  # , "twitter_information"],
        template=profile_summary_template,
        partial_variables={
            "format_instructions": person_intel_parser.get_format_instructions()
        },
    )

    """ Step-4: Chain
        1. Chains allow us to combine multiple components together to create a single, coherent application
        2. We can create a chain that takes user input, formats it with a PromptTemplate, and then passes the formatted response to an LLM. 
            -- We can build more complex chains by combining multiple chains together, or by combining chains with other components.
        3. The LLMChain is a simple chain that takes in a prompt template, formats it with the user input and returns the response from an LLM.
    """
    chain = LLMChain(llm=llm, prompt=profile_summary_prompt_template)

    """ Step-5: Before we run the llm, grab all of the input data
    """
    linkedin_profile_url = linkedin_lookup_agent(name=name)
    linkedin_profile = scrape_linkedin_profile(
        linkedin_profile_url=linkedin_profile_url
    )
    # twitter_username = twitter_lookup_agent(name=name)

    # tweets = scrape_user_tweets(username=twitter_username, num_tweets=100)

    """ Step-6: provide inputs and run
        1. Provide the necessary inputs for the prompt template.
            - Input is key-value pair.
                -- You already provided the "key" when you created PromptTemplate in Step-3
        2. run the LLM
    """
    output_from_llm = chain.run(
        linkedin_information=linkedin_profile  # , twitter_information=tweets
    )

    return person_intel_parser.parse(output_from_llm), linkedin_profile.get(
        "profile_pic_url"
    )


if __name__ == "__main__":
    print("Hello langchain")
    response_from_llm = ice_break("Christopher Shutts: master of cpq")
    print(response_from_llm)
