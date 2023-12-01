import os
from langchain.llms import OpenAI
from config import settings
from langchain import PromptTemplate


os.environ['OPENAI_API_KEY'] = settings.openai_api_key
llm = OpenAI(model_name='text-davinci-003',
             temperature=0.3,
             max_tokens=256)


def goals_proposition(list_of_goals, name, preferred_hours, hold_back, start_preference):
    goals_proposition = """Instructions:
    Acting as a coach and productivity expert, evaluate and reformulate the user-provided goals to be clearer and more actionable.
    If a goal is already well-defined, keep it as is.
    
    
    Context:
    Name: {name}
    Preferred working/studying hours: {preferred_hours}
    Common challenges: {hold_back}
    Start of day preference: {start_preference}
    
    
    User-provided goals:
    {list_of_goals}
    
    
    Output Indicator:
    Goal id: ; Goal Title: ; Goal Description: .
    
    
    Answer: """

    prompt_template = PromptTemplate(
        input_variables=["list_of_goals", "name", "preferred_hours", "hold_back", "start_preference"],
        template=goals_proposition
    )

    return llm(prompt_template.format(list_of_goals=list_of_goals, name=name, preferred_hours=preferred_hours,
                                      hold_back=hold_back, start_preference=start_preference))


def tasks_proposition(list_of_goals, list_of_tasks, name, preferred_hours, hold_back, start_preference):
    tasks_proposition = """Instructions:
    Acting as a productivity coach, analyze the provided tasks. For larger tasks, suggest breaking them down into 
    smaller actionable steps. Prioritize the tasks and spread them across a week, suggesting from 3 to 5 tasks per day.
    You can also generate tasks that might support the achievement of the provided goals.
    Tasks can be associated with a specific goal or left unassociated. Ensure tasks are actionable and relevant.
    For updated tasks use the same Task id as was in the original task, for new tasks use new Task id.


    Context:
    Name: {name}
    Preferred working/studying hours: {preferred_hours}
    Common challenges: {hold_back}
    Start of day preference: {start_preference}


    User-provided goals:
    {list_of_goals}
    
    User-provided tasks:
    {list_of_tasks}


    Output Indicator:
    Task id: ; Task Title: ; Task Description: ; Task due date: due date format = "%Y-%m-%d %H:%M:%S.%f".


    Answer: """

    prompt_template = PromptTemplate(
        input_variables=["list_of_goals", "list_of_tasks", "name", "preferred_hours", "hold_back", "start_preference"],
        template=tasks_proposition
    )

    return llm(prompt_template.format(list_of_goals=list_of_goals, list_of_tasks=list_of_tasks, name=name,
                                      preferred_hours=preferred_hours,
                                      hold_back=hold_back, start_preference=start_preference))