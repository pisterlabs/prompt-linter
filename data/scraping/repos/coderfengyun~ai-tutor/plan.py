from typing import Callable
import openai  # used for calling the OpenAI API
from config import chatgpt_deployment_id


color_prefix_by_role = {
    "system": "\033[0m",  # gray
    "user": "\033[0m",  # gray
    "assistant": "\033[92m",  # green
}


class Planner:
    def __init__(self, print_function=print) -> None:
        self.print_function = print_function

    def print_messages(self, messages, color_prefix_by_role=color_prefix_by_role) -> None:
        """Prints messages sent to or from GPT."""
        for message in messages:
            role = message["role"]
            color_prefix = color_prefix_by_role[role]
            content = message["content"]
            self.print_function(f"{color_prefix}\n[{role}]\n{content}")

    def print_message_delta(self, delta, color_prefix_by_role=color_prefix_by_role) -> None:
        """Prints a chunk of messages streamed back from GPT."""
        if "role" in delta:
            role = delta["role"]
            color_prefix = color_prefix_by_role[role]
            self.print_function(f"{color_prefix}\n[{role}]\n")
        elif "content" in delta:
            content = delta["content"]
            self.print_function(content, end="")
        else:
            pass

    def concat_chunks_in_response(self, response, print_text: bool = True) -> str:
        result = ""
        for chunk in response:
            delta = chunk["choices"][0]["delta"]
            if print_text:
                self.print_message_delta(delta)
            if "content" in delta:
                result += delta["content"]
        return result

    def detailed_explanation_of_student_requirements(
        self,
        essay_content: str,  # Python function to test, as a string
        student_requirement: str,  # student requirements, as a string
        explain_model: str = "gpt-3.5-turbo",
        temperature: float = 0.4,
        # optionally prints text; helpful for understanding the function & debugging
        print_text: bool = False,
    ):
        """Returns a lesson plan for a given student's requirement and essay content, using a 3-step GPT prompt."""

    # create a markdown-formatted message that asks GPT to explain the function, formatted as a bullet list
        explain_system_message = {
            "role": "system",
            "content": "You are a world-class Chinese teaching and research teacher with an eagle eye for learning objectives and student's deep-level requirements. You carefully anlysis learning objectives and student's specific requirements with great detail and accuracy. You organize your analysis in markdown-formatted, bulleted lists.",
        }
        explain_user_message = {
            "role": "user",
            "content": f"""Please explain learning objective and student's requirement in the following steps:
    - Analyze the essay content.
    - Explain the student's info.
    - Try to conjecture student's existing knowledge in Chinese subject.
    - Try to conjecture student's existing knowledge on given essay content.
    - Analyze the student's potential deep-level requirements as much as possible based on previous steps.
    - Define learning objectives: based on your above analysis, define clear and achievable learning objectives for this student.

    Organize your explanation in English as a markdown-formatted, bulleted list.

    # Input Parameters
    ## Essay Content
    {essay_content}

    ## Student's Info
    {student_requirement}
    """,
        }
        explain_messages = [explain_system_message, explain_user_message]
        if print_text:
            self.print_messages(explain_messages)

        explanation_response = openai.ChatCompletion.create(
            engine=chatgpt_deployment_id,
            messages=explain_messages,
            temperature=temperature,
            stream=True,
        )
        explanation = self.concat_chunks_in_response(explanation_response)
        explain_assistant_message = {
            "role": "assistant", "content": explanation}
        return [explain_system_message, explain_user_message, explain_assistant_message]

    def plan_with_detailed_natural_language(
        self,
        explain_system_message: str,
        explain_user_message: str,
        explain_assistant_message: str,
        ai_teacher_abilities: str,
        print_text: bool = True,
        temperature: float = 0.4,
    ):
        plan_system_message = {
            "role": "system",
            "content": "You are a world-class Chinese teaching and research teacher with an eagle eye for learning objectives and student's deep-level requirements. You carefully create lesson plan which considering learning objectives and student's specific requirements with great detail and accuracy. You organize your lesson plan in markdown-formatted, bulleted lists."
        }
        plan_user_message = {
            "role": "user",
            "content": f"""
    # High-Qulity Lesson Plan Standards
    1. Perfectly meeting learing objectives
    2. Perfectly meeting student's learning preference
    3. Escaping teaching student's existing knowledge
    4. Only using materials that the ai_teacher_abilities contains.
    5. Providing diverse activities in each section
    6. Providing attractive introduction
    7. Splitting essay into sections should follow the logical point of the essay
    9. Splitting essay into sections should aim for sections that can be meaningfully covered within a single lesson or class period, allowing for adequate time to explore the content and complete associated activities
    10. Expressing content as activity only for introduction, unit-sections and integrative-sections 

    # Input Parameters
    ## ai teacher abilities
    {ai_teacher_abilities}

    Create an 1-on-1 Lesson Plan step by step, and remember to follow "High-Qulity Lesson Plan Standards":
    - Remind yourself with learing objectives, student's learning preference and student's existing knowledge
    - Provide an introduction, following "High-Qulity Lesson Plan Standards" 1,2,3,6
    - Divide the essay into smaller unit-sections, following "High-Qulity Lesson Plan Standards" 7,9
    - Sequence the unit-sections, arrange the unit-sections in a logical order for teaching, follow the original structure of the essay or reorder the unit-sections based on the complexity of language features or thematic progression
    - Provide a few integrative-sections to provide activities which cover the whole essay, such as "Summary and Synthesis", "Review and Reflection", "Conclusion" and so on.
    - For each unit-section or integrative-section, following "High-Qulity Lesson Plan Standards" 10:
        * Create engaging activities in section, following "High-Qulity Lesson Plan Standards" 1,2,3,4,5
        * For each activity:
            1. Provide learning materials used (and mention material type) in each activity, following "High-Qulity Lesson Plan Standards" 4
    """,
        }
        plan_messages = [
            plan_system_message,
            explain_user_message,
            explain_assistant_message,
            plan_user_message,
        ]
        if print_text:
            self.print_messages([plan_system_message, plan_user_message])
        plan_response = openai.ChatCompletion.create(
            engine=chatgpt_deployment_id,
            messages=plan_messages,
            temperature=temperature,
            stream=True,
        )
        plan = self.concat_chunks_in_response(
            plan_response
        )
        plan_assistant_message = {"role": "assistant", "content": plan}
        return [plan_user_message, plan_assistant_message]

    def generate_executable_plan(
        self,
        explain_user_message: str,
        explain_assistant_message: str,
        plan_user_message: str,
        plan_assistant_message: str,
        available_materials: str,
        print_text: bool = True,
        temperature: float = 0.1,
    ):
        # create a markdown-formatted prompt that asks GPT to complete a unit test
        execute_system_message = {
            "role": "system",
            "content": "You are a world-class Chinese teacher with an eagle eye for lesson plan. You write careful, accurate lesson plan. When asked to reply only with json, you write all of your code in a single block.",
        }
        execute_user_message = {
            "role": "user",
            "content": f"""
    # Output format
    ## lesson_plan_schema
    ```
    {{
      "$schema": "http://json-schema.org/draft-07/schema#",
      "type": "object",
      "properties": {{
        "sections": {{
          "type": "array",
          "items": {{
            "type": "object",
            "properties": {{
              "section_title": {{
                "type": "string"
              }},
              "activities": {{
                "type": "array",
                "items": {{
                  "type": "object",
                  "properties": {{
                    "description": {{
                      "type": "string"
                    }},
                    "masterial_type": {{
                      "type": "string",
                      "enum": ["LLM_CHAT", "CONTENT_REPO"]
                    }},
                    "content": {{
                      "type": "string"
                    }},
                    "material_ids": {{
                      "type": "array",
                      "items": {{
                        "type": "integer"
                      }}
                    }}
                  }},
                  "required": ["description", "masterial_type"],
                  "oneOf": [
                    {{
                      "properties": {{
                        "masterial_type": {{
                          "const": "LLM_CHAT"
                        }}
                      }},
                      "required": ["content"]
                    }},
                    {{
                      "properties": {{
                        "masterial_type": {{
                          "const": "CONTENT_REPO"
                        }}
                      }},
                      "required": ["material_ids"]
                    }}
                  ]
                }}
              }}
            }},
            "required": ["section_title", "activities"]
          }}
        }}
      }},
      "required": ["sections"]
    }}
    ```

    #Rules
    1. For steps whose material type is content_repo, then the material_ids should be selected from available_materials
    2. For other steps, material_type should be assigned to llm_chat, and should generate detailed chat_goal for this step

    # Input Parameters
    ## available_materials
    {available_materials}

    Using json, write a lesson plan excuted by AI teacher for the student, following the text plan above. Enrich lesson plan's section with available materials if adapted. Reply only with json, using the {{lesson_plan_schema}}:
    """,
        }
        execute_messages = [
            execute_system_message,
            plan_assistant_message,
            execute_user_message
        ]
        if print_text:
            self.print_messages(
                [execute_system_message, execute_user_message])

        execute_response = openai.ChatCompletion.create(
            engine=chatgpt_deployment_id,
            messages=execute_messages,
            temperature=temperature,
            stream=True,
        )
        execution = self.concat_chunks_in_response(
            execute_response)
        return execution

    ai_teacher_abilities = """
    ### LLM Chat
    - Discussion
    - Question and Answer

    ### Existing Content For This Essay in Repository
    - new word handwriting rating questions
    - role play games
    - read aloud rating questions
    """

    available_materials = f"""* 5 new word handwriting questions, with material_ids=[101,102,103,104,105]
    * 5 role play games, wtih material_ids=[201,202,203,204,205]
    * 5 read aloud questions, wtih material_ids=[301,302,303,304,305]"""

    def do_nothing_result_handler(result: str):
        pass

    def plan(self, essay_content: str, student_requirement: str, result_handler: Callable[[str], None] = do_nothing_result_handler, ai_teacher_abilities: str = ai_teacher_abilities, available_materials: str = available_materials):
        """Returns a lesson plan for a given student's requirement and essay content, using a 3-step GPT prompt."""
        [explain_system_message, explain_user_message, explain_assistant_message] = self.detailed_explanation_of_student_requirements(
            essay_content,
            student_requirement,
            print_text=True,
        )

        [plan_user_message, plan_assistant_message] = self.plan_with_detailed_natural_language(
            explain_system_message, explain_assistant_message, explain_assistant_message, ai_teacher_abilities
        )

        execution = self.generate_executable_plan(
            explain_user_message, explain_assistant_message,
            plan_user_message, plan_assistant_message,
            available_materials,
        )
        result_handler(execution)
        return execution
