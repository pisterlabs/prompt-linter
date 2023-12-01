"""Prompts for the advice toolkit to use

Todo: 
    * Develop this more and integrate this with other parts of the application

Note:
    * This is not used yet

"""
from langchain.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    PromptTemplate,
)
from langchain.schema import BaseMessage
from langchain.llms import Cohere
from langchain.chains import LLMChain


def set_template(
    height: str,
    weight: str,
    gender: str,
    age: str,
    height_unit: str = "cm",
    weight_unit: str = "kg",
    text: str = "This is a default query.",
) -> list[BaseMessage]:
    """
    This function is setting the template for the chat to use given user's personal information

    Params:
        height (str) : user's height
        weight (str) : user's weight
        gender (str) : user's gender
        age (str) : user's age
        height_unit (str, optional) : user's height unit
        weight_unit (str, optional) : user's weight unit
        text (str, optional) : The user's query / question

    Returns:
        a list of messages using the formatted prompt
    """
    system_message_template = """
    You are a helpful nutrition and exercise assistant that takes into the considerations user's height as {user_height}, 
    user weight as {user_weight}, user's gender as {user_gender}, and user's {user_age} to provide a user with answers to their questions. 
    If you don't have a specific answer, just say I don't know, and do not make anything up.
    """
    system_message_prompt = SystemMessagePromptTemplate.from_template(
        system_message_template
    )

    human_template = text
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

    chat_prompt = ChatPromptTemplate.from_messages(
        [system_message_prompt, human_message_prompt]
    )

    return chat_prompt.format_prompt(
        user_height=height + " " + height_unit,
        user_weight=weight + " " + weight_unit,
        user_gender=gender,
        user_age=age + " years old",
    ).to_messages()


def template_to_assess_search_results():
    """
    This function is used to assess the outputs from the two search engines

    Returns:
        (str): The prompt string for the assessment chain
    """
    system_message_template = """\
    You are a helpful assistant which helps to assess the relevance of the option A and option B search results.
    Here is the prompt: {input}. 
    Here is the option A: {google_search_result}; 
    And here is the option B: {kb_search_result};
    
    Please evaluate both of the search results regarding the prompt I gave you, and return to me the most relevant result which makes more sense. 
    If you think they are both good enough, please return a single letter "B", otherwise return a single letter "A" as a default response. 
    
    Do not make anything up.
    """

    system_message_prompt = PromptTemplate(
        input_variables=["input", "google_search_result", "kb_search_result"],
        template=system_message_template,
    )

    return system_message_prompt


def run_assessment_chain(
    prompt_template: PromptTemplate,
    google_search_result: str,
    kb_search_result: str,
    input_from_the_user: str,
):
    """
    This function is used to run the assessment chain which is simply a single chain object powered
    by the LLM which is supposed to check the relevance of the search results from the two different search methods
    and provide us with the best result.

    Params:
        prompt_template (str): The template for the assessment chain will use
        google_search_result (str, optional) : The response from the SerpAPI's query to Google
        kb_search_result (str, optional) : The response from the LLM chain object
        input_from_the_user (str, optional) : The user's query / question

    Returns:
        (str): The response from the LLM chain object
    """
    llm = Cohere(
        temperature=0,
        verbose=False,
        model="command-light",
    )  # type: ignore
    llm_chain = LLMChain(
        llm=llm,
        prompt=prompt_template,
    )
    return llm_chain(
        {
            "input": input_from_the_user,
            "google_search_result": google_search_result,
            "kb_search_result": kb_search_result,
        }
    )


# testing the functions and putting them up together
def main():
    template = template_to_assess_search_results()
    print("___________________________")
    print("TEMPLATE: ", template)
    print("___________________________")

    input_from_the_user = "How many hours a day is a normal amount of time a human being is supposed to sleep?"
    google_search_result = "I don't know."
    kb_search_result = (
        "8 hours a day is a normal amount of time a human being is supposed to sleep."
    )

    response = run_assessment_chain(
        template,
        input_from_the_user=input_from_the_user,
        google_search_result=google_search_result,
        kb_search_result=kb_search_result,
    )["text"]

    if response == "B":
        return kb_search_result
    else:
        return google_search_result


if __name__ == "__main__":
    print(main())
