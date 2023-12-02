from langchain import LLMChain
from langchain.prompts.chat import (
    SystemMessagePromptTemplate,
    ChatPromptTemplate,
    HumanMessagePromptTemplate
)
from langchain.prompts import PromptTemplate

from mvp.data_manager import *
from mvp.util import remove_indent, follow_up_question_parser
from typing import *


def follow_up_question_generator(
    data_manager: DataManager,
    evaluation_manager: EvaluationManager
):
    chat_manager = ChatManager()

    prompt = ChatPromptTemplate(
        messages=[
            SystemMessagePromptTemplate.from_template(
                remove_indent(
                    f"""You are an interviewer at {data_manager.company}.

                    {data_manager.get_userdata()}
                    """)),

            HumanMessagePromptTemplate.from_template(
                remove_indent(
                    """You are an interviewer. Please read the interview question and response. If you determine that a `Follow up Question` is necessary, write the additional question you would like to ask in the areas for improvement. If you determine that it is not necessary, write `Very nice good!`. Also, please follow the format below when creating the questions:
                
                    ```
                    '심화질문':
                    - Content of follow up question
                    ```
                    And below is the interviewee's response to the interviewer's question, including the interviewer's evaluation:
                    {evaluation}
                    
                    REMEMBER! Please write in Korean.
                    REMEMBER! Please create only 1 question.
                    """))
        ],
        input_variables=["evaluation"],
    )

    followup_chain = LLMChain(llm=chat_manager.get_chat_model(),
                              prompt=prompt)
    output = followup_chain(evaluation_manager.get_answer_evaluation())
    return follow_up_question_parser(output['text'])
