import openai
import os.path
import pandas as pd
import datetime
from zoneinfo import ZoneInfo

# from helpers import get_db
from json import loads, load, dumps


BASE_DIR = os.path.dirname(os.path.abspath(__file__))


openai.api_key = "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"
messages = [
    {
        "role": "system",
        "content": "You are a intelligent assistant for a task manangement app tasked with providing personalised feedback to university students using a host of their performance metrics.",
    }
]
MESSAGE_STENCIL = """
Give a brief performance evaluation of all these metrics as a whole and how the user is doing. 
Each data point represents one day. Format the evaluation into a JSON object. Return only the JSON object, nothing else.
The JSON object should be in the same format, with all the metrics and their trends, as the following example. 
DO NOT REFER DIRECTLY TO THE GIVEN DATA POINTS AS THE USER DOESN'T HAVE ACCESS TO THEM.
Trends should be like 'Steady Increase', 'Consistent', 'Above Average' etc. 
Each object except the overall evaluation should have an 'average' key, a 'trend' key and a 'comment' key. Make sure to do this.
Make sure to be detailed in the comments and overall evaluation. Elaborate on the comments and overall evaluation.
Each metric should be a key value object where the key is one of the metrics given above. BE ABSOLUTELY CERTAIN TO MAKE EACH KEY ONE OF THE METRICS ABOVE.
Do not make any extra objects within. Only the metrics given above:

"""


def send_prompt(message):
    messages.append(
        {"role": "user", "content": message},
    )
    chat = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=messages)

    reply = chat.choices[0].message.content
    return loads(reply)


def create_message(metric_data, opening, formatting):
    return f"{opening}\n\n{metric_data}\n\n{MESSAGE_STENCIL}\n\n{formatting}"


def parse_metrics(metrics_df, handle):
    permitted_indices = [
        "total_tasks_completed",
        "daily_tasks_completed",
        "daily_busyness",
        "average_task_duration_(hours)",
        "average_hours_left_at_task_completion",
    ]
    formatted_rows = []
    formatting = []
    weekly_metrics = metrics_df.tail(7).iloc[:, 1:].T
    for index, row in weekly_metrics.iterrows():
        data_str = ", ".join(str(data) for data in row)
        formatted_rows.append(f"{index}: {data_str}")
        if index != handle and index not in permitted_indices:
            continue
        formatting.append(
            f'  "{index}": '
            + """{\n    "average": number,\n    "trend": "trend",\n    "comment": "..."\n  },"""
        )

    return (
        "\n".join(formatted_rows),
        "{\n" + "\n".join(formatting) + '\n  "overall_evaluation": "..."\n}',
    )


# def generate_new_user_report(email):
def generate_new_user_report(user_handle):
    path = os.path.join(BASE_DIR, "analytics/users")

    df = pd.read_csv(path + f"/{user_handle}.csv")
    metrics, formatting = parse_metrics(df, user_handle)
    message = create_message(
        metrics,
        f"Given the following metrics for user {user_handle}:",
        formatting,
    )
    user_reports = load(open(BASE_DIR + f"/reports/{user_handle}.json"))

    reply = send_prompt(message)
    reply["timestamp"] = datetime.datetime.now(ZoneInfo("Australia/Sydney")).strftime(
        "%Y-%m-%dT%H:%M:%S"
    )
    user_reports["user"].append(reply)

    with open(BASE_DIR + f"/reports/{user_handle}.json", "w") as output_report:
        output_report.write(dumps(user_reports, indent=2))
    return {"success": 200}


def generate_new_project_report(project_id, project_name, user_handle):
    path = os.path.join(BASE_DIR, "analytics/projects")
    df = pd.read_csv(path + f"/{project_id}.csv")
    metrics, formatting = parse_metrics(df, user_handle)
    message = create_message(
        metrics,
        f"Given the following metrics for {project_name}. Focus specifically on user {user_handle}:",
        formatting,
    )
    user_reports = load(open(BASE_DIR + f"/reports/{user_handle}.json"))

    reply = send_prompt(message)
    reply["timestamp"] = datetime.datetime.now(ZoneInfo("Australia/Sydney")).strftime(
        "%Y-%m-%dT%H:%M:%S"
    )
    if str(project_id) not in user_reports["projects"]:
        user_reports["projects"][str(project_id)] = [reply]
    else:
        user_reports["projects"][str(project_id)].append(reply)

    with open(BASE_DIR + f"/reports/{user_handle}.json", "w") as output_report:
        output_report.write(dumps(user_reports, indent=2))
    return {"success": 200}


if __name__ == "__main__":
    # generate_new_user_report("alexxu463")
    generate_new_project_report(3, "Project B", "alexxu963")
