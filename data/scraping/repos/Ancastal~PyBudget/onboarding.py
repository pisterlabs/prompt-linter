from typing import Optional
from rich.prompt import Prompt

from langchain.chains.openai_functions import (
    create_openai_fn_chain,
    create_openai_fn_runnable,
    create_structured_output_chain,
    create_structured_output_runnable,
)
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.pydantic_v1 import BaseModel, Field
import yaml
from langchain.pydantic_v1 import BaseModel, Field
from typing import List, Optional


class Expenditure(BaseModel):
    """Represents a monthly expenditure."""
    category: str = Field(..., description="The category of the expenditure")
    amount: int = Field(..., description="The amount of the expenditure")

class Person(BaseModel):
    """Identifying information about a person."""

    name: str = Field(..., description="The person's name")
    age: int = Field(..., description="The person's age")
    monthly_income: Optional[int] = Field(None, description="The person's monthly income")
    fav_color: Optional[str] = Field(None, description="The person's favorite color")
    monthly_rent: Optional[int] = Field(None, description="The person's monthly rent")
    monthly_expenditures: List[Expenditure] = Field([], description="The person's monthly expenditures with category and amount")
    current_savings: Optional[int] = Field(None, description="The person's current savings in the bank")

def onboarding():
    # If we pass in a model explicitly, we need to make sure it supports the OpenAI function-calling API.
    try:
        openai_api_key = Prompt.ask("Hello! To start, please enter your OpenAI API key")
        llm = ChatOpenAI(model="gpt-4", temperature=0, openai_api_key=openai_api_key)
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a world class algorithm for extracting information in structured formats.",
                ),
                (
                    "human",
                    "Use the given format to extract information from the following input: {input}",
                ),
                ("human", "Tip: Make sure to answer in the correct format"),
            ]
        )

        runnable = create_structured_output_runnable(Person, llm, prompt)


        from rich.console import Console
        from rich.text import Text
        from rich.panel import Panel
        from rich.markdown import Markdown

        console = Console()

        # Creating styled text
        text = Text("Hello! How are you? Tell me a bit about yourself.", style="bold green")
        text.append("\nI would like to possibly know:", style="bold blue")
        text.append("\n- Name", style="bold cyan")
        text.append("\n- Age", style="bold cyan")
        text.append("\n- Monthly Income", style="bold cyan")
        text.append("\n- Favorite Color", style="bold cyan")
        text.append("\n- Monthly Rent", style="bold cyan")
        text.append("\n- Monthly Expenditures", style="bold cyan")
        text.append("\nFinally, I would like to know your current savings.", style="bold magenta")

        # Printing within a panel for a neat and organized appearance
        console.print(Panel(text))
        print("")
        user_input = input("Enter your presentation: ")

        response = runnable.invoke({"input": user_input})
        
        with open("expenses.acct", "w") as f:
            # Extract monthly expenditures
            monthly_expenditures = response.monthly_expenditures if response.monthly_expenditures else []

            # Iterate over the expenditures and write to the file
            for expenditure in monthly_expenditures:
                category = expenditure.category.capitalize()  # Accessing attributes of the Expenditure object
                amount = expenditure.amount
                f.write(f'{category},{amount}\n')
                
        # Prepare the settings dictionary with the necessary data
        settings = {
            'monthly_rent': response.monthly_rent if response.monthly_rent else 0,
            'monthly_income': response.monthly_income if response.monthly_income else 0,
            'name': response.name,
            'age': response.age,
            'fav_color': response.fav_color if response.fav_color else "blue",
            'current_savings': response.current_savings if response.current_savings else 0,
            'openai_api_key': openai_api_key,
        }

        # Write the settings dictionary to the settings.yaml file
        with open("settings.yaml", "w") as file:
            yaml.dump(settings, file)

        print("Settings written to settings.yaml file.")
        return True
    except Exception as e:
        print(e)
        return False
    
if __name__ == "__main__":
    print("You are running the onboarding.py file directly. This is not recommended. Please run the pybudget.py file instead.")