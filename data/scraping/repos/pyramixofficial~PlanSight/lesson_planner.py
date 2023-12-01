import os

import openai
from dotenv import load_dotenv, find_dotenv


def connect_to_openai() -> openai.OpenAI: 
    """Connect to OpenAI API and return client object."""
    
    load_dotenv(find_dotenv())
    openai.api_key = os.getenv("OPENAI_API_KEY")

    client = openai.OpenAI()

    return client

# Connecting to OpenAI API client
client = connect_to_openai()


def get_grade_level_text(number: int) -> str:
    """Return the grade level text for a given number."""
    if 1 <= number <= 12:
        if number == 1:
            return "1st grade"
        elif number == 2:
            return "2nd grade"
        elif number == 3:
            return "3rd grade"
        else:
            return f"{number}th grade"
    else:
        raise ValueError("Number must be between 1 and 12")
    

def create_lesson_plan(lesson_topic: str, grade_lvl: int, lesson_time: int, language: str, subject: str) -> str:
  """Generate a lesson plan for a given topic, grade level, lesson time and language."""

  # Tranform grade level to text grade level (5 to '5th grade')
  grade_lvl_str = get_grade_level_text(grade_lvl)

  # Generate lesson plan
  response = client.chat.completions.create(
    model="gpt-4-1106-preview",
    messages=[
      {
        "role": "user",
        "content": f'''Generate a {lesson_time} minutes, {grade_lvl_str} level very descriptive lesson plan, it should be in {language} and saved as a json file, with the following structure:

- "Objectives": The learning goals for the lesson. List of points.
- "General_Competencies": The broad abilities students are expected to develop. List of points.
- "Specific_Competencies": The precise abilities related to the lesson topic. List of points.
- "Didactic_Strategies": The teaching strategies to be employed. List of points.
- "Bibliography": A list of sources referred to in the lesson. List of points.
- "Lesson_Stages": A table detailing stages of the lesson with the following columns:
    - "Lesson_Stage" (Cadrul de învățare)
    - "Teacher_Activity" (Activitatea profesorului)
    - "Student_Activity" (Activitatea elevului)
    - "Working_Time" (Timp de lucru)
    - "Methods_Procedures" (Metode și procedee)
    - "Assessment" (Evaluare)

The table should include the following rows, each representing a different stage of the lesson:
- "Organizing_Moment" (Moment organizatoric)
- "Recalling" (Evocarea)
- "Realization_of_Meaning" (Realizarea sensului)
- "Reflection" (Reflectarea)

The lesson plan should be about {lesson_topic}.
'''
      }
    ],
    response_format={"type": "json_object"},
    temperature=1,
    max_tokens=4096,
    top_p=1,
    frequency_penalty=0,
    presence_penalty=0
  )
  return response.choices[0].message.content
