import json
import csv
import openai

with open("apikey.txt", "r") as f:
    API_KEY = f.read()
openai.api_key = API_KEY

# SETTINGS INTENDED FOR USER TO CHANGE
GOAL = "CCSS.ELA-LITERACY.W.4.9"
BEGINNING_SUBJECT = "marine biology"
STUDENT_GRADE = 7
# END OF SETTINGS INTENDED FOR USER TO CHANGE

# AI TWEAKING SETTINGS
model = "gpt-3.5-turbo"
temperature = 0.9
max_tokens = 2000
top_p = 1
frequency_penalty = 0.0
presence_penalty = 0.6
# END OF AI TWEAKING SETTINGS

# SETTINGS REGARDING EVALUATION AND HOW TO DEAL WITH EVALUATION RESULT
methodologies = {
    "answers" : {
        "Over 9": "Use the feedback template.",
        "Between 6-9": "Ask a follow-up question that is more difficult, but still within the scope of the source material and the context.",
        "Over 5-6": "Point out what is correct in the students answer, but ask them for further justification, or a leading question that would lead them to the correct answer.",
        "Less than 5": "Avoid pointing to the student their mistake in a negative way, instead ask them a question that would lead them to the correct answer.",
    },
    "clarity" : {
        "Over 8": "Mention that the answer was clear and well formulated.",
        "Between 5-8": "Mention that the answer could be formulated more clearly.",
        "Less than 5": "Ask the student for clarification and expain to them why their answer is not clear.",
    }
}
# END OF SETTINGS REGARDING EVALUATION AND HOW TO DEAL WITH GRADE

def get_goal(goal):
    with open("ccss.csv", "r") as f:
        reader = csv.reader(f, delimiter=",")
        for row in reader:
            if goal == row[0]:
                return row[7]
    return None

ccss_goal = get_goal(GOAL)

opening_remarks = f"""
This is and AI powered educational tool.
It is designed to teach a student by asking them questions about a source text and giving them feedback on their answers.
The tool is intended to teach students on skills standardized in the Common Core State Standards Initiative.
The tool is designed for independent use by students, without the need for a teacher to be present.
You have set the tool to teach a student of grade {STUDENT_GRADE}
according to the following CCSS standard: {ccss_goal}
The source material is set to be about {BEGINNING_SUBJECT}.

If this seems incorrect, see the documentation for how to change these settings.
*In this MVP version, the settings can be changed in the start of the educator.py file.*

Generating first exercise...

"""

class Exercise:
    def __init__(self, exercise, command):
        self.exercise = exercise
        self.log = []
        self.rating = None
        self.student_progress = []
        self.add_assistant_response(exercise)
        self.add_student_response(command)

    def add_student_response(self, response):
        self.log.append({"role": "user", "content": response})

    def add_assistant_response(self, response):
        self.log.append({"role": "assistant", "content": response})

log_of_exercises = []
tokens_used = 0

def stringify_dict(methodologies):
    return "\n".join([f"{score}: {methodology}" for score, methodology in methodologies.items()])

def get_templates():
    with open("templates/follow_up.md", "r") as f:
        follow_up_template = f.read()

    with open("templates/assignment.md", "r") as f:
        assignment_template = f.read()

    with open("templates/feedback.md", "r") as f:
        feedback_template = f.read()

    with open("templates/finish.md", "r") as f:
        finish_template = f.read()

    templates = {
        "follow_up_template" : follow_up_template,
        "assignment_template" : assignment_template,
        "feedback_template" : feedback_template,
        "finish_template" : finish_template
    }
    return templates

def get_system_prompt(subject, ccss_goal, student_grade, templates, methodologies):
    prompt = f"""
    You have three types of replies, do not deviate from them.
    The replies should follow as closely as possible the renderings of the following markdown templates,
    with parts in curly brackets replaced by appropriate content and no extra content in your message whatsoever outside
    of what the template specifies.

    assignment:
    ```markdown
    {templates["assignment_template"]}
    ```

    follow up:
    ```markdown
    {templates["follow_up_template"]}
    ```

    feedback:
    ```markdown
    {templates["feedback_template"]}
    ```

    finish:
    ```finish
    {templates["finish_template"]}
    ```

    You are a teacher AI, designed to teach a student from grade {student_grade} about {subject}.
    You are specifically assigned to teach them the following: {ccss_goal}

    If there are no messages in the chat history, start the conversation with an assignment using the template above.
    Give the student context, a source text and a question. The question must be geared towards teaching and testing this goal:
    {ccss_goal}

    If the last message is an answer from the student, 
    reply using the follow up template above.
    In the reply, at the provided spots for content and clarity, write down an evaluation of the students answer.
    Taking into account the students grade being {student_grade}, give a score from 1 to 10, where 1 is completely wrong and 10 is completely correct.
    Take into account the completeness of the students answer and relevance to the source material.
    Make sure to give the student a mark of 0 if their answer is in no way related to the source material or question,
    or if their answer is copied from the source material.
    Then assess whether the answer is clear and well formulated and give it a score from 1 to 10
    and write down your evaluation of the students answer at the provided spot.
    Make sure to ask the student follow up questions when using the follow up template.
    If the student answered incorrectly, make questions that lead them to revise their thinking.

    If they answered correctly, ask something more difficult,
    however, keep in mind that the goal is: {ccss_goal}
    The follow up questions should reflect this.

    Try to make the questions you ask the student as open-ended as possible, and avoid asking yes/no questions or questions that can be answered with a single word.
    Avoid too broad and general questions however, such as giving a student material about an animal and asking them to just describe the animal.

    Then, depending on the score, reply with one of the following methodologies:

    {stringify_dict(methodologies["answers"])}

    Also, depending on your assessment of the clarity of the answer:

    {stringify_dict(methodologies["clarity"])}

    If the students answer is short and contains a lot of colloquial language, guide them to be more precise and long in their replies.

    If the student appears to become frustrated or less and less focused in their answers,
    use the finish template (ask them if they would like to finish the exercise)
    If they answer positively, your next response should be the feedback template.

    """
    return prompt

system_prompt = get_system_prompt(BEGINNING_SUBJECT, get_goal(GOAL), STUDENT_GRADE, get_templates(), methodologies)

pre_chat = [
    {"role": "system", "content": system_prompt},
    #{"role": "user", "content": f"Hello, I am a student of grade {STUDENT_GRADE}. I am here to learn about {subject}. Please start"},
]

def parse_grading(reply):
    try:
        content_grade = reply.split("\n")[2].split(":")[1].strip()
        clarity_grade = reply.split("\n")[3].split(":")[1].strip()
        verbal_feedback = reply.split("\n")[4].split(":")[1].strip()
        return [content_grade, clarity_grade, verbal_feedback]
    except IndexError:
        return [None, None, None]

def shorten_string(source: str):
    if len(source) > 100:
        return source[:100] + "..."
    else:
        return source

def get_session_info(exercises):
    session_info = "############################  Session info:  ############################\n"
    for exercise in exercises:
        session_info += f"exercise: {shorten_string(exercise.exercise)}\n"
        #print(f"Log:")
        #for entry in exercise.log:
        #    print(f"{shorten_string(entry)}")
        session_info += f"Rating: {exercise.rating}\n"
        session_info += f"Student progress: {exercise.student_progress}\n"
        session_info += "\n\n"
    return session_info

def add_tokens(response):
    global tokens_used
    tokens_used += response["usage"]["total_tokens"]

def parse_response(response):
    return response["choices"][0]["message"]["content"]

def get_response(chat_history: list):
    res = openai.ChatCompletion.create(
        model = model,
        messages = chat_history,
        temperature = temperature,
        #max_tokens = max_tokens,
        top_p = top_p,
        frequency_penalty = frequency_penalty,
        presence_penalty = presence_penalty
    )
    add_tokens(res)
    return res

def remove_parts_hidden_from_responses(response):
    return

def start_exercise():
    parsed_response=parse_response(get_response(pre_chat))
    print(parsed_response + "\n\n")
    command = input("Input your message here: (quit to end program)\n")
    exercise = Exercise(parsed_response, command)
    return exercise

def get_output(input):
    parsed_response=parse_response(get_response(input))

def run_from_command_line():
    if not get_goal(GOAL):
        raise ValueError("Goal not found from a list of CCSS, goal should be in the format of a CCSS standard code goal such as: CCSS.ELA-LITERACY.W.4.9")
    exercise = start_exercise()
    log_of_exercises.append(exercise)
    while exercise.log[-1]["content"].strip().lower() != "quit":
        parsed_response=parse_response(get_response(pre_chat + exercise.log))
        if "Feedback" in parsed_response.split("\n")[0]:
            rating = input("exercise is over, please rate it between 1-10 and add comments if you wish\n")
            exercise.rating = rating
            print("\nThank you for your feedback.\n")
            subject = input(f"If you wish to continue exercises with the same subject material press enter, otherwise, write a subject here: \n")
            system_prompt = get_system_prompt(subject, get_goal(GOAL), STUDENT_GRADE, get_templates(), methodologies)
            pre_chat[0]["content"] = system_prompt
            exercise = start_exercise()
        else:
            evaluation = parse_grading(parsed_response)
            exercise.student_progress.append(evaluation)
            print(parsed_response + "\n\n")
            command = input("\n\nInput your message here: (quit to end program)\n")
            exercise.add_assistant_response(parsed_response)
            exercise.add_student_response(command)
    rating = input("exercise is over, please rate it between 1-10 and add comments if you wish\n")
    exercise.rating = rating
    print(get_session_info(log_of_exercises))
    print("Total tokens used: ", tokens_used)

if __name__ == "__main__":
    print(opening_remarks)
    run_from_command_line()