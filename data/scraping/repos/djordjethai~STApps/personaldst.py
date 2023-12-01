from langchain.chat_models import ChatOpenAI
from langchain.chains import create_tagging_chain, create_tagging_chain_pydantic
from langchain.prompts import ChatPromptTemplate

from enum import Enum
from pydantic import BaseModel, Field
from langchain.chains import TransformChain, LLMChain, SimpleSequentialChain
from langchain.prompts import PromptTemplate, ChatPromptTemplate
llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-0613")


class PersonalDetails(BaseModel):
    first_name: str = Field(
        ...,
        description="This is the first name of the user.",
    )
    last_name: str = Field(
        ...,
        description="This is the last name or surname of the user.",
    )
    full_name: str = Field(
        ...,
        description="Is the full name of the user ",
    )
    city: str = Field(
        ...,
        description="The name of the city where someone lives",
    )
    email: str = Field(
        ...,
        description="an email address that the person associates as theirs",
    )
    language: str = Field(
        ..., enum=["spanish", "english", "french", "german", "italian"]
    )


user_123_personal_details = PersonalDetails(first_name="",
                                            last_name="",
                                            full_name="",
                                            city="",
                                            email="",
                                            language="")

# checking the response and adding it


def add_non_empty_details(current_details: PersonalDetails, new_details: PersonalDetails):
    non_empty_details = {k: v for k, v in new_details.dict().items() if v not in [
        None, ""]}
    updated_details = current_details.copy(update=non_empty_details)
    return updated_details

#


def check_what_is_empty(user_peronal_details):
    ask_for = []
    # Check if fields are empty
    for field, value in user_peronal_details.dict().items():
        if value in [None, "", 0]:  # You can add other 'empty' conditions as per your requirements
            print(f"Field '{field}' is empty.")
            ask_for.append(f'{field}')
    return ask_for


def ask_for_info(ask_for=['name', 'age', 'location']):

    # prompt template 1
    first_prompt = ChatPromptTemplate.from_template(
        "Below is are some things to ask the user for in a coversation way. you should only ask one question at a time even if you don't get all the info \
        don't ask as a list! Don't greet the user! Don't say Hi.Explain you need to get some info. If the ask_for list is empty then thank them and ask how you can help them \n\n \
        ### ask_for list: {ask_for}"
    )

    # info_gathering_chain
    info_gathering_chain = LLMChain(llm=llm, prompt=first_prompt)
    ai_chat = info_gathering_chain.run(ask_for=ask_for)
    return ai_chat


def filter_response(text_input, user_details):
    chain = create_tagging_chain_pydantic(PersonalDetails, llm)
    res = chain.run(text_input)
    # add filtered info to the
    user_details = add_non_empty_details(user_details, res)
    ask_for = check_what_is_empty(user_details)
    return user_details, ask_for


print(ask_for_info())

text_input = input("User: ")
user_details, ask_for = filter_response(text_input, user_123_personal_details)
while ask_for:
    ai_response = ask_for_info(ask_for)
    print(ai_response)
    text_input = input("User: ")
    user_details, ask_for = filter_response(text_input, user_details)
else:
    print(
        f'Hello, {user_details.first_name} from {user_details.city}. Everything gathered move to next phase')


