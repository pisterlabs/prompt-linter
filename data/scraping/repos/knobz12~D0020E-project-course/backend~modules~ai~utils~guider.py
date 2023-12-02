import guidance
from guidance import select, gen

@guidance()
def determineQuestionBias(lm, question: str):
    lm += f"""I want you to determine if a question for a quiz is factual or opinion based.
    For example a question about your opinion about something would be opinion based and a question about a fact is factual.

    Question: {question}

    Answer: {select(options=["factual", "opinion"])}"""
    return lm

@guidance()
def questionJSONGenerator(lm, question: str):
    lm += f"""\
    The following is a quiz question in JSON format.
    Generate four answers. Only ONE of the answers can be correct, and the other three should be incorrect.
    The three wrong answers must be different from each other but words related to the topic.

    ```json
    {{
        "question": "{question}",
        "answers": [
            {{
                "answer": "{gen("answer", stop='"')}",
                "isAnswer": "{select(options=['true', 'false'], name='isAnswer')}"
            }},
            {{
                "answer": "{gen("answer", stop='"')}",
                "isAnswer": "{select(options=['true', 'false'], name='isAnswer')}"
            }},
            {{
                "answer": "{gen("answer", stop='"')}",
                "isAnswer": "{select(options=['true', 'false'], name='isAnswer')}"
            }},
            {{
                "answer": "{gen("answer", stop='"')}",
                "isAnswer": "{select(options=['true', 'false'], name='isAnswer')}"
            }}
        ]
    }}```"""
    return lm