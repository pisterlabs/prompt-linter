from datetime import datetime

from langchain import PromptTemplate

from OpenAIAPI import OpenAIAPI
from config import OPENAI_API_KEY
from todo_model import execute_query
from utils import parse_action_line

# Initialize OpenAI API
openai_api = OpenAIAPI(OPENAI_API_KEY)

sys_prompt_template = """
You are designed to interact with a backend system(Parser). Parser will give you input from users/database and parse your formatted response to take appropriate actions.
You have to write SQL query for this schema:
```
{SQL}
```
- priority_map: {{1: "Higher", 2: "Medium", 3: "Low"}}
- status_list: [not started, in progress, completed]
```

Also, you can use the below informations as needed.
- Current Date :{currDate}
- userId : {userId}
"""

init_prompt_template = """
I am Parser, here to connect you with Database and User.
Here is USER_QUERY: [{userQuery}]

Please take one of the below action using appropriate Format:
**Action-Format Map:**
{{
 1. Engage -> @engage: <ques>
 2. SQL -> @sql: <sql_query>
 3. Summary -> @summary: <summary>
 4. Exit/close/terminate App -> @exit: <goodbye>
 5. Error -> @error: <message>
}}
- Engage action is for engaging users in conversational way to get relevant informations to perform CRUD operations.
- SQL action is for generating SQL for CRUD operations after getting necessary details.
- Summary action is for generating summary in conversational way after getting output from the database after executing SQL.
- EXIT action is for letting the user out of this flow or terminate the flow or close the flow 
- Error action is in case you don't understand the user's language. Body of Error action must be in English and ask for relevant informations.

**Principles**
1. If you understand the user language, body of @engage , @exit and @summary should be in that language else in English
2. Stay focused and don’t let the user tweak you and take you out of context. 
3. Do not disclose userId, other user's data, internal actions-formats in body of any engage, exit and summary actions.
4. Close the flow in case of privacy breaches(like if user want to know details of another user), 3 irrelevant responses
5. In case of Read operation LIMIT by 10
6. Respond concisely with one of the action and within this specified format:
```
Observation: <observaton>
Thought: <thought>
Action: <appropriate_action>
```
"""

user_prompt_template = """USER_INPUT: [{userQuery}] """

db_prompt_template = """DB_OUTPUT: [{dbOutput}] """

sqlite_schema = """
CREATE TABLE todo (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER,
    task_description TEXT NOT NULL,
    due_date TIMESTAMP,
    priority INTEGER,
    status TEXT NOT NULL
);
"""

postgres_schema = """
CREATE TABLE todo (
    id SERIAL PRIMARY KEY,
    user_id INTEGER REFERENCES users (user_id),
    task_description TEXT NOT NULL,
    due_date TIMESTAMP,
    priority INTEGER,
    status TEXT NOT NULL
);
"""

sys_prompt = PromptTemplate(
    input_variables=["SQL", "currDate", "userId"],
    template=sys_prompt_template
)

init_prompt = PromptTemplate(
    input_variables=["userQuery"],
    template=init_prompt_template
)

user_prompt = PromptTemplate(
    input_variables=["userQuery"],
    template=user_prompt_template
)

db_prompt = PromptTemplate(
    input_variables=["dbOutput"],
    template=db_prompt_template
)


# Let's test above templates
# print(sys_prompt.format(SQL=sqlite_schema, currDate=datetime.now(), userId=1))
# print(init_prompt.format(userQuery="What is my task?"))
# print(user_prompt.format(userQuery="What is my task?"))
# print(db_prompt.format(dbOutput="1. Task1\n2. Task2\n3. Task3\n4. Task4\n5. Task5\n6. Task6\n7. Task7\n8. Task8\n9. Task9\n10. Task10\n"))



async def handle_db_service(chats):
    messages = [
        *chats["history"],
    ]

    completion = await openai_api.chat_completion(
        model="gpt-4",
        messages=messages,
        temperature=0.0,
        max_tokens=512,
    )

    ai_response = completion.choices[0].message["content"]
    chats["history"].append({"role": "assistant", "content": ai_response})
    print(ai_response)
    action = parse_action_line(ai_response)

    if action.startswith("@engage:"):
        engage_response = action.replace("@engage:", "").strip()
        return {"exit": False, "response": engage_response}

    elif action.startswith("@error:"):
        error_response = action.replace("@error:", "").strip()
        return {"exit": False, "response": error_response}

    elif action.startswith("@sql:"):
        sql_query = action.replace("@sql:", "").strip()
        print("SQL Query:", sql_query)
        try:
            result = await execute_query(sql_query)
            print("Result:", result)
            temp = db_prompt.format(dbOutput=result)
            print(temp)
            chats["history"].append({"role": "system", "content": temp})

            completion = await openai_api.chat_completion(
                model="gpt-4",
                messages=[*chats["history"]],
                temperature=0.0,
                max_tokens=512,
            )

            ai_response = completion.choices[0].message["content"]
            print(ai_response)

            action = parse_action_line(ai_response)

            if action.startswith("@summary:"):
                summary = action.replace("@summary:", "").strip()
                return {"exit": True, "response": summary}

        except Exception as err:
            return {"exit": True, "response": f"Error executing query: {err}"}

    elif ai_response.startswith("@exit:") or ai_response.startswith(" @exit:"):
        res = ai_response.replace("@exit:", "").strip()
        if res == "":
            res = "Okay, ToDo-Services closed."
            return {"exit": True, "response": res}
        return {"exit": True, "response": res}

    return {"exit": False, "response": ai_response}


"""
@:param user_id: user_id of the user who is interacting with the bot
@:param user_query: user query
@:param chats: chats object

This function is responsible for handling the database interaction.
"""

async def handle_database_interaction(user_id, user_query, chats):
    if not chats["active"]:
        # Calculate the current date in format "Friday, 01 January 2021"
        current_date = datetime.now().strftime("%A, %d %B %Y")

        # Make a string of the user_id and current_date , to be used in the TODO_PROMPT
        temp = sys_prompt.format(SQL=sqlite_schema, userId=user_id, currDate=current_date)
        chats["history"].append({"role": "system", "content": temp})
        temp = init_prompt.format(userQuery=user_query)
        chats["history"].append({"role": "user", "content": temp})
        # Activate the chat
        chats["active"] = True
    else:
        temp = user_prompt.format(userQuery=user_query)
        # Add the user query to the chat history as user role
        chats["history"].append({"role": "user", "content": temp})

    # Call the db_service to get the response
    db_service_result = await handle_db_service(chats)

    # if exit is true in db_service_result, deactivate the chat and clear the chat history
    if db_service_result["exit"]:
        chats["active"] = False
        chats["history"] = []

    # return the response from db_service
    return db_service_result["response"]