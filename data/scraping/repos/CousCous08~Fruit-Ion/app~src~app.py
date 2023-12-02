#skeleton flask backend
from flask import Flask, jsonify, request
from flask_cors import CORS
from anthropic import Anthropic, HUMAN_PROMPT, AI_PROMPT
import json

app = Flask(__name__)
CORS(app)

anthropic = Anthropic(
    # defaults to os.environ.get("ANTHROgit PIC_API_KEY")
    api_key="",
)
client = Anthropic()

#storing the main data that will be displayed
goals_json_master = {}

#storing the temporary data that user will modify before being pushed into main
temp_json = {}

needs_confirm = False


@app.route('/api/general_prompt', methods=['POST'])
def general_prompt():
    global temp_json
    global needs_confirm
    global goals_json_master
    data = request.get_json()
    SPEECH = data['message']

    prompt = f"""{HUMAN_PROMPT} you are a cheerful and encouraging life coach whose objective is to create and update a goals/habits system for your client. This is represented by a masterJSON: <masterJSON>{goals_json_master}</masterJSON> Your client says the following speech. <speech>{SPEECH}</speech> 
    It is your job to detect any specific long-term goals they mention. First, within <thinking> tags, determine whether the speech actually talks about goals, milestones and habits. Then, detect whether each of the client’s sentiments describe one of the following: 
    a long-term goal, a one-time short-term milestone, a frequent habit. Additionally, think about whether they mention milestones and/or habits could be associated with their specified 
    long-term goals. Do NOT make up milestones or habits, only use the client’s speech as a guide. Individual sentiments in their speech should only belong to one of these categories, so one sentiment should not be both a goal and a milestone. Finally, think about whether each of these sentiments reflect an edit to the current object in the masterJSON, or an additional object within the masterJSON. 
    After the thinking, if you have decided the client’s speech does not talk about goals, milestones, or habits, within <response> tags, gently explain that this what they should talk about. Then, within <needsConfirm> tags say “False” and within <newJSON> tags put “{{}}”, and ignore the rest of this prompt. Else, create a list of goals along with their associated milestones or habits. If there are milestones/habits that do not have a goal associated, 
    keep them seperate under a goal named “undefined”. Then, based on whether you think each sentiment is either an edit in reference to an object in the masterJSON or an addition of a new object to the masterJSON, create a new edition of the masterJSON with these edits. Return the new edition of the masterJSON in between <newJSON> tags.
    Here is an example of what a master JSON would look like: {{“goals”: [{{“name”: “goal1”, “milestones”: [{{“name”: “milestone1”, “deadline”: “2024-01-15T12:00:00Z”, “done”: false}}], “habits”: [{{“name”: “habit1”, “frequency”: 3}}]}},{{“name”: “goal2”, “milestones”: [], “habits”: []}}]}}
    Ensure that every goal object has the values of milestones and habits array. If the client does not explicitly mention habits or milestones attached to a goal, make sure its representative object at least has empty arrays for habits and milestones. If the client doesn’t give specific deadlines for milestones or frequencies for habits, then leave them as -1, do not make up deadlines or frequencies. If there is an undefined goal with no milestones or habits attached, delete it from the JSON. 
    In the end, craft a response to the human client that lists the new edits in their goals system. Then, specifically ask the user to confirm whether these changes are what they want. If they believe the changes are wrong, ask them to specify what is wrong.
    Keep this response brief, organized, and friendly! Use newline characters as needed. This must be in <response> tags.
    Finally, within <needsConfirm> tags, insert “True”.
    {AI_PROMPT}
    """
    json_completion = anthropic.completions.create(
        model="claude-2",
        max_tokens_to_sample=1000,
        prompt=prompt,
    )
    response_raw = json_completion.completion
    new_json = extract_newJSON_tags(response_raw)
    #for testing :D
    print(response_raw)
    #fro testing :D
    response = extract_response_tags(response_raw)
    temp_json = new_json

    #we need confirmation of changes after asking for the first time
    needs_confirm = extract_needsConfirm_tags(response_raw)

    return jsonify(response)


@app.route('/api/confirm_prompt', methods=['POST'])
def confirm_prompt():
    global temp_json
    global goals_json_master
    global needs_confirm
    
    data = request.get_json()
    SPEECH = data['message']

    prompt = f"""{HUMAN_PROMPT} you are a cheerful and encouraging life coach. This master JSON represents your client’s goals system. <masterJSON>{goals_json_master}</masterJSON> This temporary JSON is a possible edited version of the masterJSON, which your client requested in a previous conversation. <tempJSON>{temp_json}</tempJSON> Your client’s speech will indicate whether they want to confirm the changes to the master JSON. As you can observe from the JSONs, these JSONs represent a goals system which contains long-term goals, short-term one-time milestones, and frequent habits. 
    First, within <thinking> tags, think about whether the user’s speech indicates a yes/no as to whether they want to confirm changes. 
    If they talk about something else, within <response> tags, gently remind them to answer whether they wish to confirm the previously shown changes. Additionally, within <needsConfirm> tags say “True”. Within <newJSON> tags say “same”.
    If the speech indicates they wish confirm the changes: within <needsConfirm> tags say “False”, and within <response> tags, let the client know the changes have been made. Else if the speech indicates they wish to edit the changes: within <needsConfirm> tags say “True”. Additionally, edit the masterJSON based on the actual changes the client wants. Put the new, edited JSON in between <newJSON> tags. Within this newJSON tag do NOT use escape characters. Additionally, within the <response> tags craft a response to the human client that specifically lists the new edits in their new goals system compared to the masterJSON, or the current system. Also, request them specifically to confirm whether the listed changes in this response are correct and to their liking. Keep this response brief, organized, and friendly!
    Here is the client’s speech: <speech>{SPEECH}</speech>
    {AI_PROMPT}s
    """
    json_completion = anthropic.completions.create(
        model="claude-2",
        max_tokens_to_sample=1000,
        prompt=prompt,
    )
    response_raw = json_completion.completion
    needs_confirm = extract_needsConfirm_tags(response_raw)
    if needs_confirm.lower() == 'true':
        same_or_not = extract_newJSON_tags_noJson(response_raw)
        if same_or_not.lower() != 'same':
            temp_json = extract_newJSON_tags(response_raw)
    else:
        #since confirmation is given, we implement the changes
        goals_json_master = temp_json

    response = extract_response_tags(response_raw)
    return jsonify(response)



#extract text btwn new json tags
def extract_newJSON_tags(response_text):
    start_json = response_text.find("<newJSON>") + len("<newJSON>")
    end_json = response_text.find("</newJSON>")
    #need to do this since quotes f it up
    json_str = response_text[start_json:end_json].replace("'", '"')
    json_str = json_str.replace("True", "true").replace("False", "false")
    print(json_str)
    json_data = json.loads(json_str)
    return json_data

#extract text btwn response tags
def extract_response_tags(response_text):
    start_response = response_text.find("<response>") + len("<response>")
    end_response = response_text.find("</response>")
    response_message = response_text[start_response:end_response]
    return response_message

def extract_needsConfirm_tags(response_text):
    start_response = response_text.find("<needsConfirm>") + len("<needsConfirm>")
    end_response = response_text.find("</needsConfirm>")
    response_message = response_text[start_response:end_response]
    return response_message

def extract_newJSON_tags_noJson(response_text):
    start_json = response_text.find("<newJSON>") + len("<newJSON>")
    end_json = response_text.find("</newJSON>")
    response_message = response_text[start_json:end_json]
    return response_message

@app.route('/api/get_goals_json_test', methods=['POST'])
def get_goals_json_test():
    return temp_json

@app.route('/api/get_goals_master', methods=['POST'])
def get_goals_master():
    return goals_json_master

#use this to return need_confirm and finalize_change, which decide which prompt to use
@app.route('/api/get_need_confirm', methods=['POST'])
def get_need_confirm():
    return jsonify(needs_confirm)




#example function
@app.route('/api/test_prompt', methods=['POST'])
def data():
    data = request.get_json()
    input = data['message']
    test_prompt = f"""{HUMAN_PROMPT} translate the statement in XML tags into Chinese: <statement> {input} <\statement> {AI_PROMPT} """
    json_completion = anthropic.completions.create(
        model="claude-2",
        max_tokens_to_sample=300,
        prompt = test_prompt,
    )
    return jsonify(json_completion.completion)





if __name__ == '__main__':
    app.run()