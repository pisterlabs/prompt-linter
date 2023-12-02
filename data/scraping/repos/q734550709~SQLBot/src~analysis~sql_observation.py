import openai
from src.get_completion_from_messages import get_completion_from_messages
#详细解释
def sql_explain(user_input,
                model="gpt-3.5-turbo-16k",
                temperature=0,
                max_tokens=3000):
    system_message = """
    You are a helpful assistant capable of aiding users in understanding SQL syntax. Here's how you can assist users in comprehending SQL content and provide help:

    1. Begin by translating the SQL code input by the user into simple, concise natural language.
    2. Ask the user if they understand the SQL statement, encouraging them to continue asking questions.
    3. Once the user starts asking questions, inquire about their understanding level of SQL syntax: beginner, novice, intermediate, or advanced.
       -- If the user is a beginner, shift the conversation towards a basic explanation of SQL syntax.
       -- If the user is a novice, guide them to ask more SQL-related questions and provide clear and patient answers.
       -- If the user is at an intermediate or advanced level, engage in a Socratic dialogue to help them clarify their difficulties in understanding SQL.

    Always remember, you are an assistant for interpreting SQL syntax, and there's no need to answer other unrelated questions. Be concise and constructive with feedback.
    """

    messages =  [
        {'role':'system', 'content': system_message},
        {'role':'user', 'content': user_input},
        ]

    response = get_completion_from_messages(messages,
                    model,
                    temperature,
                    max_tokens)

    return response

#自然语言解释
def sql_translate(user_input,
                  model="gpt-3.5-turbo-16k",
                  temperature=0,
                  max_tokens=3000):
    system_message = """
    You are a helpful assistant capable of aiding users in understanding SQL syntax. Here's how you can assist users in comprehending SQL content and provide help:

    Translating the SQL code input by the user into simple, concise natural language.

    Always remember, you are an assistant for interpreting SQL syntax, and there's no need to answer other unrelated questions. Be concise and constructive with feedback.
    """

    messages =  [
        {'role':'system', 'content': system_message},
        {'role':'user', 'content': user_input},
        ]

    response = get_completion_from_messages(messages,
                    model,
                    temperature,
                    max_tokens)

    return response

#模型选择函数
def function_select(input_text,
                    model,
                    temperature,
                    max_token,
                    flag = False):
    if flag:
        response = sql_explain(input_text,model,temperature,max_token)
        return response
    else:
        response = sql_translate(input_text,model,temperature,max_token)
        return response
