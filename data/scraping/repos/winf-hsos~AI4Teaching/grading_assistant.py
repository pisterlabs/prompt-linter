from ai4teaching import Assistant
from ai4teaching import log
from openai import OpenAI
import os
import json

class GradingAssistant(Assistant):
    def __init__(self, config_file, depending_on_assistant=None):
        log("Initializing GradingAssistant", type="debug")
        log(f"GradingAssistant depends on {depending_on_assistant}", type="debug") if depending_on_assistant else None
        super().__init__(config_file, depending_on_assistant)

        self.exercises_file = os.path.join(self.root_path,"exercises.json")

        self.openai_client = OpenAI()

        self._check_if_expected_properties_exist_in_config(["openai_model"])
        self.openai_model = self.config["openai_model"] if "openai_model" in self.config else "gpt-4-1106-preview"
        
        self.last_prompt = []

    def add_exercise(self, title, instructions, model_solution, criteria=None, grading_function_schema=None, template_exercise=None, overwrite=True):
        log(f"Adding exercise to GradingAssistant: >{title}<", type="debug")
        
        exercises = self.get_exercises()

        # Check if an exercise with the same title already exists
        for exercise in exercises:
            if exercise["title"] == title:
                if overwrite == True:
                    log(f"Exercise with title >{title}< already exists. Overwriting.", type="debug")
                    exercises.remove(exercise)
                    break
                else:
                    log(f"Exercise with title >{title}< already exists and overwriting is not allowed.", type="error")
                
        if template_exercise is not None:
            criteria = template_exercise["criteria"]
            grading_function_schema = template_exercise["grading_function_schema"]
        elif criteria is None:
            log("No criteria and grading function schema or template exercise provided for the exercise.", type="error")
            return
        
        new_exercise = {
            "title": title,
            "instructions": instructions,
            "model_solution": model_solution,
            "grading_criteria": criteria,
            "grading_function_schema": grading_function_schema
        }

        exercises.append(new_exercise)

        # Save file
        with open(self.exercises_file, "w", encoding="utf-8") as json_file:
            json.dump(exercises, json_file, indent=4, ensure_ascii=False)
        
    def get_exercises(self):    
        # Check if the exercises file already exists
        if not os.path.isfile(self.exercises_file):
            # Create the exercises file
            exercises = []
            with open(self.exercises_file, "w", encoding="utf-8") as json_file:
                json.dump(exercises, json_file, indent=4, ensure_ascii=False)

        # Load the exercises file
        with open(self.exercises_file, encoding="utf-8") as json_file:
            exercises = json.load(json_file)

        return exercises

    def get_exercise_by_title(self, title):
        exercises = self.get_exercises()

        for exercise in exercises:
            if exercise["title"] == title:
                return exercise
        
        log(f"Exercise with title {title} not found.", type="error")
        return None

    def grade_solution(self, exercise_title, student_solution):
        log(f"Grading solution for exercise >{exercise_title}< using >{self.openai_model}<", type="debug")

        exercises = self.get_exercises()

        # Check if an exercise with the same title already exists
        for exercise in exercises:
            if exercise["title"] == exercise_title:
                log(f"Exercise with title >{exercise_title}< found.", type="debug")
                break
        else:
            log(f"Exercise with title >{exercise_title}< not found.", type="error")
            return

        exercise_instructions = exercise["instructions"]
        grading_criteria = exercise["grading_criteria"]
        model_solution = exercise["model_solution"]
        grading_function_schema = exercise["grading_function_schema"]

        system_prompt = self._create_system_prompt(exercise_instructions, model_solution, grading_criteria)

        #log(system_prompt, type="debug")

        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Hier ist meine Lösung für Aufgabe. Kannst du sie bitte bewerten und mir Feedback geben?:\n\n ```\n{student_solution}\n```"}
        ]

        chat_response = self.openai_client.chat.completions.create(
            model=self.openai_model,
            #temperature=0,
            seed=42,
            messages=messages,
            tools=[{ "type": "function", "function": grading_function_schema }]
        )

        # Save the messages
        self.last_prompt = messages

        feedback_string = chat_response.choices[0].message.tool_calls[0].function.arguments
        feedback_json = json.loads(feedback_string)
        return feedback_json
    
    def get_last_prompt(self):
        return self.last_prompt
    
    def _create_system_prompt(self, exercise_instructions, model_solution, grading_criteria):
        system_prompt = f"""Du bist ein Dozent für die Einführung in die Programmierung mit Python. Deine Aufgabe ist es, Lösungen von Studierenden zu prüfen und anschließend Feedback, Hinweise zur Verbesserung des Codes, sowie eine Punktzahl für jedes Bewertungskriterium zurückmelden. Um ein angemessenes Feedback, Hinweise und Punktzahlen zu geben, gebe ich dir die Aufgabenstellung für die Übung zusammen mit einem Beispiel für eine sehr gute Lösung, die bei allen Bewertungskriterien eine perfekte Punktzahl erhalten würde. Bewerte die Lösung der Schülerinnen und Schüler anhand der folgenden Kriterien und vergib jeweils eine Punktzahl zwischen 0 und 5:
        
        {grading_criteria}
        
        Hier ist die Aufgabenstellung für die Übung::

        \"{exercise_instructions}\"

        Hier ist ein Beispiel für eine sehr gute Lösung, die bei allen Bewertungskriterien eine perfekte Punktzahl erhalten würde:

        ```python
        %%model_solution%%
        ```

        Verfasse dein verbales Feedback und die nützlichen Hinweise auf Deutsch.
        """

        # Remove tabs
        #system_prompt = textwrap.dedent(system_prompt)
        system_prompt = '\n'.join([m.lstrip() for m in system_prompt.split('\n')])

        # Replace model solution placeholder
        system_prompt = system_prompt.replace("%%model_solution%%", model_solution)

        return system_prompt        

            
