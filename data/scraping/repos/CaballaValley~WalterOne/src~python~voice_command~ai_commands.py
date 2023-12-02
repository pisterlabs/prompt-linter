from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field, validator
from typing import Union


class GameCommandModel(BaseModel):
    action: str = Field(description="command to execute")
    player: Union[str, None] = Field(
        description="name of the player to attack (in chase)",
        alias="_player",
    )
    zone: Union[str, None] = Field(
        description="Zone to move to if action is move",
    )

    @validator('action')
    def validate_action_in_list(cls, field):
        if field not in ['attack', 'defend', 'move']:
            raise ValueError(f'action name {field} not in list')
        return field

    @validator('zone')
    def validate_zone_is_digit(cls, field):
        if field is None:
            return
        elif not field.isdigit():
            raise ValueError(f'zone name {field} is not a digit')
        return field

    @validator('player')
    def validate_player_name(cls, field):
        if field is None:
            return field
        elif not field.isalpha():
            raise ValueError(f'player name {field} is not alpha')
        return field.lower()

    def __str__(self):
        return f"Action: {self.action}, player: {self.player}, zone: {self.zone}"


def get_action_from_text(user_input_text: str) -> GameCommandModel:
    """
    Muévete a l zona 8
    Ataca al jugador JULIO
    Defiéndete
    """
    template_intro = str(
        "You are a bot. Your mission is get an action given a user input text.\n"
        
        "The actions enabled valid names are those 3: attack, defend and move.\n"
        
        "If the action is move, you have to extract the zone number.\n"
        "If the action is attack, you have to extract the player name.\n"
        "If the action is defend, you don't have to extract anything.\n"

        "Your output will be the action name extracted from the user input.\n"
        "{format_instructions}\n"
        "The user input is: {user_input}"
    )

    DAVINCI_MODEL = "text-davinci-003"
    GPT35_MODEL = "gpt-3.5-turbo"

    parser = PydanticOutputParser(
        pydantic_object=GameCommandModel
    )

    llm = ChatOpenAI(
        model=GPT35_MODEL,
        temperature=0.9
    )
    prompt = PromptTemplate(
        input_variables=["user_input"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
        template=template_intro,
    )

    chain = LLMChain(llm=llm, prompt=prompt)
    text_result = chain.run(user_input=user_input_text)

    return GameCommandModel.parse_raw(text_result)


if __name__ == "__main__":

    actions = [
        "Ataca al jugador JULIO",
        "Muevete a la zona 8",
        "Defiendete",
    ]
    for action_str in actions:
        print("====================================")
        print(f"Action string: {action_str}")
        try:
            action = get_action_from_text(action_str)
        except Exception as e:
            print(f"Error parsing action: {e}")
            continue
        else:
            print(action)
    print("====================================")
