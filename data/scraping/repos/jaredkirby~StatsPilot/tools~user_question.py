from langchain.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)


def get_user_question_answer(chat, user_input: str, level: int):
    sys_template = "You are a helpful Statistics expert assisting a student further their understanding."
    sys_message_prompt = SystemMessagePromptTemplate.from_template(sys_template)
    question_template = f"""\
    Question:
    {user_input}
    Please answer in such a way a {level} grade level student can best understand your explination.
    Format your response in markdown.
    """
    question_message_prompt = HumanMessagePromptTemplate.from_template(
        question_template
    )
    answer_prompt = ChatPromptTemplate.from_messages(
        [
            sys_message_prompt,
            question_message_prompt,
        ]
    )
    formatted_answer_prompt = answer_prompt.format_prompt(
        user_input=user_input
    ).to_messages()
    llm = chat
    result = llm(formatted_answer_prompt)
    return result.content
