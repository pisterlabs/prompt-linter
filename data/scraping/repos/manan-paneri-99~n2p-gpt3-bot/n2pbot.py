import os
import openai
from dotenv import load_dotenv


load_dotenv()

openai.api_key = os.environ.get("OPENAI_KEY")

completion = openai.Completion()

start_sequence = "\nPoints for report:"
restart_sequence = "\nNotes:"

start_chat_log = '''Convert meeting notes of half sentences to grammatically correct agenda points in the meeting summary.
\n\n
Notes: Silverlie Project- John Doe explains the model of data pipeline. Issues- Data inconsistency identified. Discussion: Changes suggested, Dev team acknowledges. To-do- Implementation deadline by Monday. Add to next meet.
\n
Point for report: As per the next agenda of the meeting on Silverline project, John Doe explained the model of the data pipeline. He identified the issue of data inconsistency in the model. In the discussion, changes were suggested to the Dev team and the suggestions were acknowledged. The deadline for implementing the suggestions for the presented problem was assigned to be Monday. Added to the next meeting's agenda.
\n\n
Notes: Sandbox app- Rachael pitches new scan and upload feature for Sandbox app. Discussion- Suggestions requested, Mark suggests using auto-capture. Action- Suggestions noted and motion passed. To-do- update next meet.
\n
Points for report: As per the next meeting agenda on the Sandbox app, Rachael pitched a new scan and upload feature in the app. She requested for suggestions to improve the feature. Mark suggested using auto-capture as the scanning method. Rachael noted Mark's suggestions. The motion for the scan and upload feature was passed. Updating on the suggested improvements was added to the next meet's agenda.
\n\n
'''


def convert(notes, chat_log=None):
	if chat_log is None:
		chat_log = start_chat_log

	prompt = f'{chat_log}Notes: {notes}\nPoints for report:'

	response = completion.create(
		engine="davinci", prompt=prompt, temperature=0.45, max_tokens=200, top_p=1, frequency_penalty=0, presence_penalty=0, 
		stop=["\n", "Notes:", "Points for report:"])

	point = response.choices[0].text.strip()
	return point


def append_interaction_to_chat_log(notes, point, chat_log=None):
	if chat_log is None:
		chat_log = start_chat_log
	return f'{chat_log}Notes: {notes}\n Points for report: {point}\n'
