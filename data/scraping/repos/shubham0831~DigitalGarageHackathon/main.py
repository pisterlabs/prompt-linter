meeting_notes = '''
Here are the results for the tickets mentioned in the meeting:

jira_ticket_number: CLOUD-467
ticket_description: Enhance user profile UI and add recent activity section
user: shubhampareek4000
action_item: Nothing to do
previous_release_version: 4.34
suggested_release_version: 4.34  
previous_story_points: 5
suggested_story_points: 5
reasoning: On track for release

jira_ticket_number: CLOUD-46
ticket_description: Implement image uploads for user profiles
user: dmitriybaikov  
action_item: Monitor external API integration
previous_release_version: 4.34
suggested_release_version: 4.34
previous_story_points: 5 
suggested_story_points: 7
reasoning: Facing delays due to Imgur API downtime which is critical for image uploads. Increasing story points to account for external dependency.

jira_ticket_number: CLOUD-47
ticket_description: Integrate machine learning capabilities using TensorFlow  
user: dmitriybaikov
action_item: Break into subtasks focusing on data preprocessing first
previous_release_version: 4.34
suggested_release_version: 4.35 
previous_story_points: 5
suggested_story_points: 8
reasoning: More complex than expected due to model training and compatibility issues with TensorFlow. Breaking into subtasks and pushing to next release.

jira_ticket_number: CLOUD-48
ticket_description: Not provided
user: dmitriybaikov  
action_item: Blocked due to third-party API outage
previous_release_version: 4.34
suggested_release_version: 4.35
previous_story_points: 5
suggested_story_points: 5 
reasoning: Dependent on external API so pushing to next release

jira_ticket_number: JIRA-127
ticket_description: Not provided
user: shubhampareek4000
action_item: Nothing to do  
previous_release_version: 4.34
suggested_release_version: 4.34
previous_story_points: 5
suggested_story_points: 5
reasoning: Back on track after resolving requirements 

jira_ticket_number: JIRA-128
ticket_description: Improve user profile page UI 
user: shubhampareek4000
action_item: Nothing to do
previous_release_version: 4.34 
suggested_release_version: 4.34
previous_story_points: 5 
suggested_story_points: 5
reasoning: On track for release

jira_ticket_number: JIRA-129 
ticket_description: Implement profile image uploads  
user: shubhampareek4000
action_item: Nothing to do
previous_release_version: 4.34
suggested_release_version: 4.34
previous_story_points: 5
suggested_story_points: 3
reasoning: Straightforward feature, reducing story points.

jira_ticket_number: JIRA-130
ticket_description: Optimize database queries using Hibernate
user: shubhampareek4000
action_item: Nothing to do
previous_release_version: 4.34
suggested_release_version: 4.34
previous_story_points: 5
suggested_story_points: 8 
reasoning: Critical for performance, complex due to optimizing queries with Hibernate

jira_ticket_number: JIRA-131  
ticket_description: Implement client-side caching using Redis
user: shubhampareek4000
action_item: Nothing to do 
previous_release_version: 4.34
suggested_release_version: 4.34
previous_story_points: 5
suggested_story_points: 8
reasoning: Critical for performance, complex due to caching implementation with Redis 

jira_ticket_number: JIRA-132
ticket_description: Address critical bugs reported by QA
user: shubhampareek4000
action_item: Nothing to do
previous_release_version: 4.34
suggested_release_version: 4.34
previous_story_points: 5
suggested_story_points: 5  
reasoning: Critical bugs but uncertainty around root causes. 

jira_ticket_number: JIRA-133
ticket_description: Fix usability issue in login flow
user: dmitriybaikov 
action_item: Nothing to do
previous_release_version: 4.34  
suggested_release_version: 4.34
previous_story_points: 5 
suggested_story_points: 3
reasoning: Less complex compared to other bugs

jira_ticket_number: JIRA-134
ticket_description: Integrate new payment gateway using Stripe  
user: dmitriybaikov
action_item: Set up contingency plan for delays
previous_release_version: 4.34
suggested_release_version: 4.34
previous_story_points: 5
suggested_story_points: 10
reasoning: Complex integration due to multiple payment scenarios with Stripe.

In summary, the key highlights from this meeting are:

- Most tickets are on track for 4.34 release. JIRA-125 and JIRA-126 pushed to 4.35 due to dependencies. 

- Critical performance optimization tasks JIRA-130 and JIRA-131 identified and estimated at 8 points each.

- JIRA-124 and JIRA-134 require monitoring of external API integrations. Contingency plans set up.

- JIRA-125 broken into subtasks to simplify machine learning integration.

- Story points adjusted based on task complexity like JIRA-129 reduced to 3 points.

Overall, the team provided estimates for new tasks, identified risks and dependencies to streamline the sprint execution.
'''

from anthropic import Anthropic, HUMAN_PROMPT, AI_PROMPT
from slack_sdk import WebClient
import re
import json
from jira import JIRA



def generate_prompt(prompt_dict) -> str:
    generated_prompt = "Here is a conversation between a human and you, go through it and analyze it, and then based on the context, response to the human question"
    index = 0

    while index in prompt_dict.keys():
        convo = prompt_dict[index]

        question = convo['prompt']
        question = "human : " + question
        generated_prompt = generated_prompt + "\n" + question


        response = convo['reply']
        if response != "":
           response = "ai : " + response
           generated_prompt = generated_prompt + "\n" + question

        index+=1
    
    return generated_prompt, index-1


ai_behavior_prompt = '''
    I will be sending you a transcript from a meeting. Based on this transcript I want you to do the following:

    List all the tickets mentioned in this meeting, and for each ticket, give me the following output:

        jira_ticket_number : ticket_number,
        ticket_description : what is the original ticket about
        user : the name of the user this ticket belongs to,
        action_item : action item (eg. increase in story points, delayed release, new subtasks, etc)
        previous_release_version: release version of the task (4.34 by default)
        suggested_release_version: release version you suggest
        previous_story_points: story point of the ticket (default value is 5)
        suggested_story_points: story points you suggest
        reasoning: reasoning for the change
        

        For story points, if a task is taking longer than expected suggest an increase in the story points. Assume each task has a story point of 5, then based on how difficult it is proving
        update the story point by 1,3,5, or 7

        Assume each task is due for 4.34 release. If it seems like it will get delayed, push the release back to 4.35.

        Each ticket can only belong to one user, and not multiple user, identify that user. It is usually the person who first gives updates on the task.

        If a ticket has no action item, just put action item as "nothing to do"

        If you suggest change for any story, provide your reasoning as well, in the reasoning section. Elaborate on this reasoning and don't be vague. If a person has mentioned a reason
        about something being complex, figure out what in particular is complex and put that in reasoning.

    Once you have done this for each ticket, give me a high level summary of this meeting.

    Just reply ok to this message, I will send the transcript in the next message.
    '''

# Create a Jira client instance
jira = JIRA(server=JIRA_SERVER, basic_auth=(JIRA_USERNAME, JIRA_API_TOKEN))

def update_story_point(ticket_number, new_story_points):
    int_story_points = int(new_story_points)
    print(f"Updating story points for {ticket_number} to {new_story_points}")
    issue = jira.issue(ticket_number)
    # for field_name in issue.raw['fields']:
    #     field_value = issue.raw['fields'][field_name]
    #     print(f'{field_name}: {field_value}')
    issue.update(fields={"customfield_10016": int_story_points})

meeting_transcript_file= open("/Users/shubham/Code/personal/DigitalGarageHackathon/sample_meeting_transcript.txt", "r")
meeting_transcript = meeting_transcript_file.read()
meeting_transcript_file.close()

previousPrompts = {
    0 : {
        "prompt" : ai_behavior_prompt,
        "reply" : "ok"
    }, 
    1 : {
        "prompt" : f"Here's the transcript from the meeting. Please do the tasks I mention : \n {meeting_transcript}",
        "reply" : ""
    },
}

prompt, last_index = generate_prompt(previousPrompts)

# completion = anthropic.completions.create(
#      model="claude-2",
#      max_tokens_to_sample=1000000,
#      prompt=f"{HUMAN_PROMPT} {prompt} {AI_PROMPT}",
#  )

# previousPrompts[last_index]['reply'] = completion.completion

# print(completion.completion)

client = WebClient(token=SLACK_TOKEN)

result = client.users_list()
members = result.data['members']

split_notes = meeting_notes.split("\n\n")
ticket_dicts = []

keys_to_extract = [
    'user',
    'jira_ticket_number',
    'ticket_description',
    'action_item',
    'previous_release_version',
    'suggested_release_version',
    'previous_story_points',
    'suggested_story_points',
    'reasoning'
]

for section in split_notes:
    ticket_dict = {}
    for key in keys_to_extract:
        match = re.search(fr'{key}: (.+)', section)
        if match:
            ticket_dict[key] = match.group(1)
    ticket_dicts.append(ticket_dict)

for ticket in ticket_dicts:
    if 'suggested_story_points' in ticket and ticket['action_item'] != 'Nothing to do':
        username = ticket['user']
        str_representation = json.dumps(ticket)

        if ticket['jira_ticket_number'] == "CLOUD-46":
            print(ticket)
        # for m in members:
        #     if m['name'] == username:
        #         user_id = m['id']
        #         # sent message to user, check for response in the future, for now just update the ticket
        #         client.chat_postMessage(channel=user_id, text=str_representation)
        #         # print(m)
        #         if ticket['jira_ticket_number'] == "CLOUD-46" or ticket['jira_ticket_number'] == "CLOUD-47":
        #             update_story_point(ticket['jira_ticket_number'], ticket['suggested_story_points'])



