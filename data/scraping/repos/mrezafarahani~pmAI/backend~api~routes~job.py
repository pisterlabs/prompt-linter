# import requests
# from dateutil.parser import parse
# import openai
# import os

# # Set up OpenAI API key
# openai.api_key = os.getenv('OPENAI_API_KEY')
# TRELLO_API_BASE_URL = 'https://api.trello.com/1/'

# def get_trello_board_tickets(trello_token, board_id):
#     url = f'{TRELLO_API_BASE_URL}boards/{board_id}/cards'
#     params = {'key': trello_token, 'fields': 'name,due'}
#     response = requests.get(url, params=params)
#     response.raise_for_status()  # Raise an exception if the request failed

#     tickets = response.json()
#     return tickets

# def schedule_meeting(tickets):
#     # Generate the agenda using the ChatGPT API
#     print("In schedule Metting")
#     print(tickets)
#     response = openai.Completion.create(
#         engine="text-davinci-003",
#         prompt=f'I need to schedule a sprint planning meeting for these Trello tickets. Can you help me draft an agenda for this meeting? Tickets are {"\n".join(tickets)}',
#         temperature=0.5,
#         max_tokens=100
#     )

#     agenda = response.choices[0].text.strip()

#     print(f'Meeting Agenda:\n{agenda}')

# def start_job(trello_token, board_id, participants):
#     tickets = get_trello_board_tickets(trello_token, board_id)
#     schedule_meeting(tickets)
#     # for ticket in tickets:
#     #     ticket_name = ticket['name']
#     #     due_date_str = ticket.get('due')
#     #     if due_date_str is not None:
#     #         due_date = parse(due_date_str)
#     #         schedule_meeting(ticket_name, due_date, participants)
