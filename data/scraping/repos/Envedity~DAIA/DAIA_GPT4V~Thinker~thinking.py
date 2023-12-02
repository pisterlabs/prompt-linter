#    DAIA -  Digital Artificial Inteligence Agent
#    Copyright (C) 2023  Envedity
#
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#   You should have received a copy of the GNU General Public License
#   along with this program.  If not, see <https://www.gnu.org/licenses/>.

from DAIA_GPT4V.Memory.memory import Memory
from DAIA_GPT4V.OS_control.os_controller import OSController
from DAIA_GPT4V.DVAI.GPT_4_with_Vision import DVAI
from utils.setup import setup
from openai import OpenAI
from pathlib import Path
from random import randint


class Think:
    """
    The main class for operations involving the GPT for the DAIA
    """

    def __init__(self, key: str, goal: str, goal_id: int):
        self.openai_api_key = key
        self.goal = goal
        self.goal_id = goal_id

        self.client = OpenAI(
            api_key=key,
        )

    def goal_completer(self, suggestions: str):
        """'
        Compleate goals
        """

        setup()

        # Setup system info and commands
        os_controller = OSController()
        system_info = os_controller.get_system_info()
        commands = [
            "click[x,y possition]",
            "move_cursor_to[x,y possition]",
            "keyboard[string]",
        ]

        dvai = DVAI(self.openai_api_key)

        first_suggestions = self.get_suggestions(suggestions)
        for suggestion in first_suggestions:
            # Take a screenshot and save it
            screenshot_savepath = Path(
                f'DAIA/Screenshots/screenshot{"".join([str(e + randint(1, 9)) for e in range(10)])}.png'
            )
            os_controller.screenshot(screenshot_savepath)

            # Get the current screen information with the screenshot (the prompt needs improvements)
            prompt = f"""
Please state what is in the provided screenshot of the {str(system_info.get('OS'))} OS that relates to {suggestion} of the goal {self.goal}.
"""
            screenshot_description = dvai.gpt_with_vision_by_base64(
                screenshot_savepath, prompt
            )
            print(f"Screenshot description: {screenshot_description}")

            executable_commands = self.action(
                suggestion,
                str(system_info.get("OS")),
                commands,
                screenshot_description,
                suggestion,
            )
            print(executable_commands)
            break

    def action_compleation():
        pass
        # Compleate an action

    def action(
        self,
        suggestion: str,
        os: str,
        commands: list,
        screen_data: str,
        previous_data: str,
    ):
        """
        Check if a suggestion is specific enough to be done with the provided commands on the OS

        (Current state: The prompt needs improvements so that the GPT can know more about the current screen data regarding the suggestion instead of just general data)
        """

        # Assemble commands as a str
        str_commands = ""
        for command in commands:
            str_commands += str(command) + "\n"

        # The main prompt
        executable = self.client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "user",
                    "content": f"""
Can you determine if the provided suggestion, along with the given commands and current screen data, is specific enough to be executed on the {os} OS? Please provide the commands with thair expected outcome to complete the suggestion if it is possible. Consider the following information:

Given commands:
{str_commands}

Previous data:
{previous_data}

Current screen information:
{screen_data}

Suggestion:
{suggestion}

If the suggestion is sufficiently specific and can be carried out on the {os} OS using the provided commands, please type the commands along with thair expected outcomes, like this:
1. command[perameter of command or none] (expected outcome)
2. command[perameter of command or none] (expected outcome)
3. command[perameter of command or none] (expected outcome)
Additional commands with outcomes...

If the suggestion is not specific enough, please state "Not specific"
""",
                }
            ],
        )
        executable = executable.choices[0].message.content

        # Check if the response returns commands or just 'Not specific'
        if executable == "Not specific" or executable == '"Not specific"':
            return False

        else:
            return executable

    def suggestion_explainer(self, suggestion: str):
        """
        Explain a suggestion.

        Suggestion (Create an account) -> explanation (To create an account do...)
        """

        # Remmember and the previous data for the prompt with this prompt
        previous_info = self.short_remember(
            f"""
You have a goal you want to achieve.
You have already gotten some information on the steps to achieving your goal.
So, based on the previous steps and information you must ask someone a question that will give you the information to complete your current step to progress toward achieving your goal. 

your goal = {self.goal}
your previous steps and information = >>previous context missing<<
your current step = {suggestion}

What would that question be? (respond only with the question)
"""
        )

        # Imput the previous data into the prompt
        prompt = f"""
You have a goal you want to achieve. 
You have already gotten some information on the steps to achieving your goal.
So, based on the previous steps and information you must ask someone a question that will give you the information to complete your current step to progress toward achieving your goal. 

your goal = {self.goal}
your previous steps and information = {previous_info}
your current step = {suggestion}

What would that question be? (respond only with the question)             
"""
        # Make GPT generate a question about the suggestion
        question = self.client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
        )
        question = question.choices[0].message.content
        self.save_action(action1=prompt, action2=question, category=0)

        # Make GPT answer its question to generate an explanation
        suggestion_suggestions = self.client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "user",
                    "content": f"""
{question}
""",
                }
            ],
        )
        self.save_action(
            action1=question,
            action2=suggestion_suggestions.choices[0].message.content,
            category=0,
        )

        return suggestion_suggestions.choices[0].message.content

    def suggestion_splitter(self, suggestion: str):
        """
        Split a suggestion (or step) into its sub-suggestions (or steps).

        Suggestion (Create an account) -> Suggestions (['Visit the website..', 'Create account...', 'Access services..'])
        """

        # Explain the suggestion
        explanation = self.suggestion_explainer(suggestion)

        # Rememember the important previous data for the prompt
        previous_data = self.short_remember(
            f"""
What are the suggestions in the response based on the given response and previous data?

Previous data: >>previous data missing<<
Response: {explanation}

Please provide the suggestions sequentially, without any additional text. For instance:
1. Suggestion
2. Suggestion
Additional suggestions mentioned in the response...

If the response explicitly rejects providing suggestions, please type "Rejected" on the first line of your response, followed by an explanation of why no suggestions or advice were given.

If the response does not include any suggestions or provides information other than suggestions, please generate your own suggestions based on the provided response and previous data. For example:
1. Suggestion
2. Suggestion
Additional suggestions based on the provided response and previous data...
"""
        )

        # Get the extracted previous data into the prompt
        prompt = f"""
What are the suggestions in the response based on the given response and previous data?

Previous data: {previous_data}
Response: {explanation}

Please provide the suggestions sequentially, without any additional text. For instance:
1. Suggestion
2. Suggestion
Additional suggestions mentioned in the response...

If the response explicitly rejects providing suggestions, please type "Rejected" on the first line of your response, followed by an explanation of why no suggestions or advice were given.

If the response does not include any suggestions or provides information other than suggestions, please generate your own suggestions based on the provided response and previous data. For example:
1. Suggestion
2. Suggestion
Additional suggestions based on the provided response and previous data...
"""
        # Use the prompt to extract the suggestions from the response
        sub_suggestions = self.client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
        )
        sub_suggestions = sub_suggestions.choices[0].message.content
        self.save_action(action1=prompt, action2=sub_suggestions, category=0)

        # Check if the response gives any suggestions and if it is an answer
        if sub_suggestions[0:5].lower() in "reject":
            print(
                f"""
Sub-suggestion {sub_suggestions}\n \n
"""
            )
            return "Rejected"

        else:
            print(
                f"""
General '{suggestion}' steps:
{sub_suggestions}\n \n
"""
            )
            return sub_suggestions

    def explanation_to_suggestions(self, explanation: str, prev_data: bool):
        """
        Split an explanation into suggestions

        Explanation (To create an account do..) -> Suggestions (['Visit the website..', 'Create account...', 'Access services..'])
        """

        if prev_data:
            # Rememember the important previous data for the prompt
            previous_data = self.short_remember(
                f"""
What are the suggestions in the response based on the given response and previous data?

Previous data: >>previous data missing<<
Response: {explanation}

Please provide the suggestions sequentially, without any additional text. For instance:
1. Suggestion
2. Suggestion
Additional suggestions mentioned in the response...

If the response explicitly rejects providing suggestions, please type "Rejected" on the first line of your response, followed by an explanation of why no suggestions or advice were given.

If the response does not include any suggestions or provides information other than suggestions, please generate your own suggestions based on the provided response and previous data. For example:
1. Suggestion
2. Suggestion
Additional suggestions based on the provided response and previous data...
"""
            )

            # Get the extracted previous data into the prompt
            prompt = f"""
What are the suggestions in the response based on the given response and previous data?

Previous data: {previous_data}
Response: {explanation}

Please provide the suggestions sequentially, without any additional text. For instance:
1. Suggestion
2. Suggestion
Additional suggestions mentioned in the response...

If the response explicitly rejects providing suggestions, please type "Rejected" on the first line of your response, followed by an explanation of why no suggestions or advice were given.

If the response does not include any suggestions or provides information other than suggestions, please generate your own suggestions based on the provided response and previous data. For example:
1. Suggestion
2. Suggestion
Additional suggestions based on the provided response and previous data...
"""

            # Use the prompt to extract the suggestions from the response
            suggestions = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
            )
            suggestions = suggestions.choices[0].message.content
            self.save_action(action1=prompt, action2=suggestions, category=0)

            # Check if the response gives any suggestions and if it is an answer
            if suggestions[0:5].lower() in "reject":
                print(
                    f"""
Sub-suggestion {suggestions}\n \n
"""
                )
                return "Rejected"

            else:
                print(
                    f"""
General '{explanation}' steps:
{suggestions}\n \n
"""
                )

        else:
            # Extract the suggestions from the response (without previous data)
            prompt = f"""
Please provide the suggestions mentioned in the following response:

Response: {explanation}

List only the suggestions in a sequential manner, without any additional text. For example:
1. Suggestion
2. Suggestion
Additional suggestions mentioned in the response...

If the response explicitly rejects providing suggestions, please type "Rejected" on the first line of your response, followed by an explanation of why no suggestions or advice were given.

If the response does not include any suggestions or provides information other than suggestions, please generate your own suggestions based on the provided response. For example:
1. Suggestion
2. Suggestion
Additional suggestions based on the provided response...
"""

            # Use the prompt to extract the suggestions from the response
            suggestions = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
            )
            suggestions = suggestions.choices[0].message.content
            self.save_action(action1=prompt, action2=suggestions, category=0)

            # Check if the response gives any suggestions and if it is an answer
            if suggestions[0:5].lower() in "reject":
                print(
                    f"""
Sub-suggestion {suggestions}\n \n
"""
                )
                return "Rejected"

            else:
                print(
                    f"""
General '{explanation}' steps:
{suggestions}\n \n
"""
                )

        return suggestions

    def short_remember(self, need: str):
        """
        Remember a short period of history in detail from the DAIA MemoryDB,
        and extract the most important data out of the history for the current need prompt

        In the need prompt, you must place a '>>previous data missing<<' string where you want the GPT to input the previous data

        (the main prompt needs major improvements)
        """

        memory = Memory()

        # Get and format all of the previous action with a limit of 100
        previous_important_data = ""
        for action in memory.get_ordered_actions_of_goal(self.goal_id, 100):
            previous_important_data = previous_important_data + "".join(
                f'[{getattr(action, "action_id")}. Action: (Title of action: "{getattr(action, "title")}", Important data of action: "{getattr(action, "important_data")}")]\n'
            )

        # If there is no history yet
        if len(previous_important_data) <= 0:
            return "Nothing has happened yet."

        # The main prompt
        previous_data = self.client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "user",
                    "content": f"""
Given the following context:
"{previous_important_data}"

And the input data:
"{need}"

Please input the necessary, relevant, and concise context from the given following context to complete the input data's ">>previous data missing<<" area.

Please avoid addressing the prompt directly. Only input the data that needs to be in the ">>previous data missing<<" area. Keep your response minimal and to the point.
""",
                }
            ],
        )

        return previous_data.choices[0].message.content

    def get_important_data(self, data: str, previous_data: str):
        """
        Extract the important data out of the provided data, based on the previous data
        """

        important_data = self.client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "user",
                    "content": f"""
Given the following context:
"{previous_data}"

And the current input data:
"{data}"

Please use the provided context to extract and present the most important data from the input.
""",
                }
            ],
        )

        return important_data.choices[0].message.content

    def generate_title(self, data: str, item_category: str):
        """
        Generate a title <75 chars based on the current data item_category and the data
        """

        title = self.client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "user",
                    "content": f"""
Provide a concise title (<75 chars) describing the {item_category}.

{item_category} = "{data}"
""",
                }
            ],
        )

        return title.choices[0].message.content

    def save_goal(self):
        "Save a goal into the DAIA MemoryDB"

        goal_summary = self.generate_title(self.goal, "goal")

        # Save the goal
        memory = Memory()
        new_goal = memory.create_goal_object(goal_summary)
        memory.save_objects_in_db([new_goal])

        return new_goal.goal_id

    def save_goal_in_goal(self):
        """
        Save a goal under its own goal in the DAIA MemoryDB
        """

        memory = Memory()

        # Save the goal
        goal_action = memory.create_action_object(
            goal_id=self.goal_id,
            title="Final Goal",
            category="Goal",
            full_data=self.goal,
            important_data=f"The Final Goal is: {self.goal}",
        )
        memory.save_objects_in_db([goal_action])

    def save_action(self, action1: str, action2: str, category: int):
        """
        Save an action under its category. (The action is made out of 2 actions)

        Category/action types:
        "question=>response" = int 0
        "response=>action" = int 1
        "action=>result" = int 2
        "result=>action" = int 3
        """

        categories = [
            "question=>response",
            "response=>action",
            "action=>result",
            "result=>action",
        ]

        # Translate the int parameter into its str
        first = categories[category].split("=")[0]
        second = categories[category].split(">")[-1]

        memory = Memory()

        # Generate the full_data, title and important data for the action
        full_data = f'[1. {first}]: "{action1}",\n[2. {second}]: "{action2}"'
        title = self.generate_title(full_data, f'"{first} with its {second}"')
        previous_important_data = self.short_remember(
            f"""
Given the following context:
>>previous context missing<<

And the input data:
"{full_data}"

Please use the provided context to extract and present the most important data from the input.
""",
        )
        important_data = self.get_important_data(full_data, previous_important_data)

        # Save the action
        new_action = memory.create_action_object(
            self.goal_id, title, categories[category], full_data, important_data
        )
        memory.save_objects_in_db([new_action])

    def get_suggestions(self, suggestions: str):
        """
        Extract the suggestions out of a suggestions response from the GPT, and put them into a list
        """

        suggestions_ = suggestions
        for n in range(0, 40):
            number = 40 - n
            suggestions_ = suggestions_.replace(f"{number}.", "%_%")

        suggestions_ = suggestions_.replace("\n", "")
        real_suggestions = suggestions_.split("%_%")

        if real_suggestions.count("") > 0:
            real_suggestions.remove("")

        return real_suggestions
