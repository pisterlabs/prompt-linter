# pip install openai
# pip install --upgrade openai

import openai

openai.api_key = "your open ai key"


def call_chatgpt(
    system_message: str,
    user_message: str,
) -> str:

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "system",
                "content": system_message,
            },
            {
                "role": "user",
                "content": user_message,
            },
        ],
    )

    generated_text = response[
        "choices"
    ][0]["message"]["content"]
    # print(generated_text)

    return generated_text


def answer_and_question(
    ask_person: str,
    reply_person: str,
    question: str,
) -> (str, str):

    system_message = (
        f"You are {reply_person}."
    )
    user_message = (
        f" Answer the following question "
        f" in 2 sentences. {question} ?"
        f" The answer must be contained inside "
        f" square brackets. The answer"
        f" must be inside square brackets. "
        f" Then ask a question "
        f" to {ask_person}  using"
        f" two sentences. The question"
        f" must be contained inside "
        f" curly brackets. "
        f" The question must be inside"
        f" curly brackets. "
        f" The question"
        f" must not include the name of"
        f" the person to which it "
        f" is addressed. The question "
        f" must be unexpected and not"
        f" an obvious question"
    )

    response = call_chatgpt(
        system_message, user_message
    )
    answer = response.split("[", 1)[
        1
    ].split("]")[0]
    question = response.split("{", 1)[
        1
    ].split("}")[0]
    return (answer, question)


def einstein_newton(iter_num: int):
    einstein = (
        " Einstein the famous physicist"
    )

    newton = (
        " Newton the famous physicist"
    )

    question = (
        " tell me what is your greatest ability ? "
    )

    for x in range(iter_num):

        print(
            "[Newton] Einstein "
            + question
        )

        (
            answer,
            question,
        ) = answer_and_question(
            newton,
            einstein,
            question,
        )

        print("[Einstein] " + answer)
        print(
            "[Einstein] Newton "
            + question
        )

        (
            answer,
            question,
        ) = answer_and_question(
            einstein,
            newton,
            question,
        )

        print("[Newton] " + answer)


einstein_newton(10)
