from langchain import PromptTemplate
from langchain.chat_models import ChatVertexAI
from langchain.chains import LLMChain

from output_parsers import summary_parser, ice_breaker_parser, topics_of_interest_parser

chat_model = ChatModel.from_pretrained("chat-bison@001")
    parameters = {
        "temperature": 0.8,
        "max_output_tokens": 1024,
        "top_p": 0.8,
        "top_k": 40,
    }
chat = chat_model.start_chat(  # Initialize the chat with model
        # chat context and examples go here
)

def get_summary_chain() -> LLMChain:
    summary_template = """
         given the information about a person from linkedin {information}, and twitter posts {twitter_posts} I want you to create:
         1. a short summary
         2. two interesting facts about them
         \n{format_instructions}
     """

    summary_prompt_template = PromptTemplate(
        input_variables=["information", "twitter_posts"],
        template=summary_template,
        partial_variables={
            "format_instructions": summary_parser.get_format_instructions()
        },
    )

    return LLMChain(llm=chat_model, prompt=summary_prompt_template)


def get_interests_chain() -> LLMChain:
    interesting_facts_template = """
         given the information about a person from linkedin {information}, and twitter posts {twitter_posts} I want you to create:
         3 topics that might interest them
        \n{format_instructions}
     """

    interesting_facts_prompt_template = PromptTemplate(
        input_variables=["information", "twitter_posts"],
        template=interesting_facts_template,
        partial_variables={
            "format_instructions": topics_of_interest_parser.get_format_instructions()
        },
    )

    return LLMChain(llm=chat_model, prompt=interesting_facts_prompt_template)


def get_ice_breaker_chain() -> LLMChain:
    ice_breaker_template = """
         given the information about a person from linkedin {information}, and twitter posts {twitter_posts} I want you to create:
         2 creative Ice breakers with them that are derived from their activity on Linkedin and twitter, preferably on latest tweets
        \n{format_instructions}
     """

    ice_breaker_prompt_template = PromptTemplate(
        input_variables=["information", "twitter_posts"],
        template=ice_breaker_template,
        partial_variables={
            "format_instructions": ice_breaker_parser.get_format_instructions()
        },
    )

    return LLMChain(llm=chat, prompt=ice_breaker_prompt_template)
