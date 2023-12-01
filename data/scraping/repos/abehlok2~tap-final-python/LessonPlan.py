import os
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.schema import SystemMessage
from prompts.lp_prompt import lp_prompt
from prompts.lp_system_msg import lp_system_msg

openai_api_key = os.getenv("OPENAI_API_KEY")

gpt4 = ChatOpenAI(
    openai_api_key=openai_api_key,
    temperature=0.5,
    streaming=True, 
)


class LessonPlan:
    """
    This class represents a Lesson Plan that uses LangChain's LLMChain to generate 
    and modify lesson plans.
    """
    lp_system_msg = lp_system_msg

    lp_chat_prompt = ChatPromptTemplate.from_messages([lp_system_msg, lp_prompt])

    def __init__(self,
                 subject,
                 topic,
                 learning_objective,
                 additional_details,
                 lp_chat_prompt,
                 memory=None):
        """
        Initialize a new instance of LessonPlan.
        """
        self.subject = subject
        self.topic = topic
        self.learning_objective = learning_objective
        self.additional_details = additional_details

        # Use provided lp_chat_prompt and memory or default values
        self.lp_chat_prompt = lp_chat_prompt
        self.memory = memory

        self.lp_gen_chain = LLMChain(
            llm=gpt4,
            prompt=self.lp_chat_prompt,
            memory=self.memory,
        )

    def generate_lesson_plan(self):
        lp = self.lp_gen_chain.predict(
            subject=self.subject,
            topic=self.topic,
            learning_objective=self.learning_objective,
            additional_details=self.additional_details)
        return lp

    def modify_lesson_plan(self, memory, mods: str):
        """Modify a lesson plan to meet user-requested changes"""
        mod_sys_msg = SystemMessage(content="""
        Please make the necessary requested modifications
        to the lesson plans, as specified by the user. Ask if the new lesson plan is 
        acceptable once you have completed the modification process. 
        """)
        human_mod_prompt = "Please go back and make the following modifications to the \
            lesson plan: {mods}"

        input_variables = ["mods"]

        mod_prompt_template = HumanMessagePromptTemplate.from_template(
            template=human_mod_prompt, input_variables=input_variables)

        mod_chat_prompt = ChatPromptTemplate.from_messages([mod_sys_msg,
                                                            mod_prompt_template])
        mod_chain = LLMChain(
            llm=gpt4,
            prompt=mod_chat_prompt,
            memory=memory,
        )

        modified_lp = mod_chain.run(mods)
        return modified_lp

    @staticmethod
    def save_lessonplan(self, lessonplan, path):
        """Save a lesson plan to a lessonplan_docx file"""
        lessonplan_doc = lessonplan.Document()
        lessonplan_doc.add_heading(lessonplan.subject, 0)
        lessonplan_doc.add_heading(lessonplan.topic, 1)
        lessonplan_doc.add_heading(lessonplan.learning_objective, 2)
        lessonplan_doc.add_heading(lessonplan.additional_details, 3)
        if not os.path.exists(path):
            print("Save location does not exist!")
            print("Would you like to create a new folder? (y/n)")
            if input("> ").lower() == "y":
                os.mkdir(path)
                print("Folder created!")
            else:
                print("Exiting...")
                return
        lessonplan_doc.save(path)