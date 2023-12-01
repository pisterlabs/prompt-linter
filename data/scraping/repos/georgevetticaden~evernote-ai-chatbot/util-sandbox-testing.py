from datetime import datetime, timezone

from langchain.memory import ConversationBufferMemory
from langchain.llms import OpenAI
from langchain.chains import ConversationChain

import ast

import re


# string_of_tuples= """('When did my elbow pain occur?', 'The right elbow pain occurred on January 11, 2023.\nSOURCES: /Users/aju/Dev-Testing/evernotes//Family_ Aju_ Health_Right Elbow Pain - 01-11-23_77bd1ddc-a2f4-ff71-f982-c8e00b99c8ff.enex'), ('What was the cause and what did the doctor suggest?', 'The cause of the right elbow pain is elbow bursitis. The doctor suggested seeing an orthopedic to extract the fluid and test, and if it happens again, to come in within 24 hours to see if they can extract the fluid. The doctor also suggested waiting for 3 more months and seeing if the pain goes away with therapy before considering a cortisone shot or something else. \nSOURCES: /Users/aju/Dev-Testing/evernotes//Family_ Aju_ Health_Right Elbow Pain - 01-11-23_77bd1ddc-a2f4-ff71-f982-c8e00b99c8ff.enex, /Users/aju/Dev-Testing/evernotes//Family_ Aju_ Health_Appointment with Dr. Patel on 7_11 for Right Elblow Pain_670413eb-c656-96cd-4a60-57e57faf2372.enex, /Users/aju/Dev-Testing/evernotes//Family_ Aju_ Health_Meeting with Dr. Patel - Ortho - 6-28-22_2aa4a604-8676-72d3-e6c7-1e95d3218bc6.enex, /Users/aju/Dev-Testing/evernotes//Family_ Aju_ Health_CAll with Dr. Patel on Elbow - 1_31_23_41485090-e28e-0d1f-e759-b06618097958.enex'), ('Where did i go for physical therapy for the pain?', 'The physical therapy location recommended for the right elbow pain is Achieve Physical Therapy located at 1315 Macon Drive, Suite 105, Naperville, IL 60564. \nSOURCES: /Users/aju/Dev-Testing/evernotes//Family_ Aju_ Health_Physical Therapy for Elbow_dc5108b8-a13e-935a-8b82-3379af4b600a.enex, /Users/aju/Dev-Testing/evernotes//Family_ Aju_ Health_Appointment with Dr. Patel on 7_11 for Right Elblow Pain_670413eb-c656-96cd-4a60-57e57faf2372.enex'), ('What is the address?', 'The address for Achieve Physical Therapy recommended for right elbow pain is 1315 Macom Drive, Suite 105, Naperville, IL 60564. \nSOURCES: /Users/aju/Dev-Testing/evernotes//Family_ Aju_ Health_Physical Therapy for Elbow_dc5108b8-a13e-935a-8b82-3379af4b600a.enex, /Users/aju/Dev-Testing/evernotes//Family_ Aju_ Health_Meeting with Dr. Patel - Ortho - 6-28-22_2aa4a604-8676-72d3-e6c7-1e95d3218bc6.enex')"""
# # string_of_tuples='()'
# escaped_string = string_of_tuples.replace('\n', '-' )
#
# print(escaped_string)
#
#
# array_of_tuples = ast.literal_eval('[' + escaped_string + ']')
#
# print(len(array_of_tuples))
#
# print(array_of_tuples)
#
# print("********************************")
#
# # print(array_of_tuples[0])
# # print("********************************")
# # print(array_of_tuples[1])
# # print("********************************")
# # print(array_of_tuples[2])
#
#

# date_string = "2023-06-21 12:00:00"
#
#
# dt_object = datetime.strptime(date_string, "%Y-%m-%d %H:%M:%S")
# timestamp = int(dt_object.timestamp())
#
# revert = datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d %H:%M:%S")
# print(revert)


# current_dtm = datetime.now()
# current_timestamp = int(current_dtm.timestamp())
#
# current_date_string = datetime.fromtimestamp(current_timestamp).strftime("%Y-%m-%d %H:%M:%S")
# print(current_date_string)


# timestamp_int = 1687457368
# print( datetime.fromtimestamp(timestamp_int).strftime("%Y-%m-%d %H:%M:%S"))

# query0="What can you help with?"
# answer = "Answer to query0"
#
# chat_history = []
# chat_history.append((query0, answer))
# print(chat_history)


chat_history ='''("What can you help with?", "I\'m a helpful assistant for George answering his questions based on the notes within his Evernote. Let me know if there is anything else I can do at this time or if a call is in order.        SOURCES: /Users/aju/Dev-Testing/evernotes//Family_ Aju_Better Parthav_87866f75-4f4b-4541-abb5-e53858ba3d50.enex, /Users/aju/Dev-Testing/evernotes//Finance_ Financial Advisor_Instructions from Josh about Tax Docs and where to find Info_86b41750-f154-40c7-903d-f720d0cecccd.enex, /Users/aju/Dev-Testing/evernotes//House-Reminders_Set up Tournament bracket_fb3df5f7-4b67-49c1-a1f8-1c16f273ccd0.enex, /Users/aju/Dev-Testing/evernotes//House-Reminders_Apply weed killer_774549f5-637c-4466-9ca6-d44c2872ed7a.enex") ("When did my right elbow pain occur?", "Your right elbow pain occurred on January 11, 2023.SOURCES: /Users/aju/Dev-Testing/evernotes//Family_ Aju_ Health_Right Elbow Pain - 01-11-23_77bd1ddc-a2f4-ff71-f982-c8e00b99c8ff.enex, /Users/aju/Dev-Testing/evernotes//Family_ Aju_ Health_Appointment with Dr. Patel on 7_11 for Right Elblow Pain_670413eb-c656-96cd-4a60-57e57faf2372.enex, /Users/aju/Dev-Testing/evernotes//Family_ Aju_ Health_Meeting with Dr. Patel - Ortho - 6-28-22_2aa4a604-8676-72d3-e6c7-1e95d3218bc6.enex, /Users/aju/Dev-Testing/evernotes//Family_ Aju_ Health_CAll with Dr. Patel on Elbow - 1_31_23_41485090-e28e-0d1f-e759-b06618097958.enex.") ("What doctors did i see for it?", "You saw Dr. Dsilva for your right elbow pain and she gave you a needle steroid after the oral steroid of prednisone didn\'t really help. However, she couldn\'t extract the fluid because it was too thick. She wants you to see an orthopedic to extract the fluid and test. Dr. Patel suggested waiting for 3 more months and seeing if the pain goes away with therapy for tennis elbow. If not, he suggested a cortisone shot or something else. He also suggested a specialized place called Achieve for therapy. For your wrist, Dr. Patel took another x-ray and ordered an MRI to see what the next step could be. One next step could be cutting 1mm bone off which is longer, and this procedure is called tenex. Dr. Disilva recommended taking a high 12-day dosage of Prednisone (Steroids) and not taking diclofenac to decrease pain/inflammation. If the steroids improved, then potentially take steroids instead of diclofenac which was helping. The problem with steroids is that it might increase AC1/Diabetes. The problem with taking daily diclofenac is that it affects kidney levels. You need to talk to Dr. Toby on the Diabetes and Dr. Sujith on the kidney stuff.        SOURCES: /Users/aju/Dev-Testing/evernotes//Family_ Aju_ Health_Right Elbow Pain - 01-11-23_77bd1ddc-a2f4-ff71-f982-c8e00b99c8ff.enex, /Users/aju/Dev-Testing/evernotes//Family_ Aju_ Health_Meeting with Dr. Patel - Ortho - 6-28-22_2aa4a604-8676-72d3-e6c7-1e95d3218bc6.enex, /Users/aju/Dev-Testing/evernotes//Family_ Aju_ Health_Appointment with Dr. Patel on 7_11 for Right Elblow Pain_670413eb-c656-96cd-4a60-57e57faf2372.enex, /Users/aju/Dev-Testing/evernotes//Family_ Aju_ Health_Dr. Disilva Yearly Visit - 5-15-23_b49ac509-a30f-ff38-af4e-8eaaccd62f78.enex")'''
# escaped_chat_history = chat_history.replace('\n', '-')
#
# array_of_tuples_chat_history = ast.literal_eval('[' + escaped_chat_history + ']')


# # Use regex to extract tuples from the chat_history string
# regex_pattern = r'\((.*?)\)'
# array_of_tuples_chat_history = re.findall(regex_pattern, chat_history)
#
# # Split each tuple string into individual values
# array_of_tuples_chat_history = [tuple_str.split('", "') for tuple_str in array_of_tuples_chat_history]
#
# # Create tuples by stripping the quotes and whitespace from each value
# array_of_tuples_chat_history = [(value[1:-1], value2[1:-1]) for value, value2 in array_of_tuples_chat_history]
#
# # Print the array of tuples
# for tuple_item in array_of_tuples_chat_history:
#     print(tuple_item)

source = '''/Users/aju/Dev-Testing/evernotes//Notebook__Family_ Aju_ Health__Note__Appointment with Dr. Patel on 6_20 for Right Elbow Pain__Id__2372.enex, /Users/aju/Dev-Testing/evernotes//Notebook__Family_ Aju_ Health__Note__Right Elbow Pain - 01-11-23__Id__c8ff.enex'''
pattern = r"Notebook__(.*?)__Note__(.*?)__Id__(.*?)\.enex"
matches = re.findall(pattern, source)

# Create a list of formatted strings
formatted_strings = []
for match in matches:
    notebook_value = match[0]
    note_value = match[1]
    formatted_string = f"Notebook: {notebook_value}, Title: {note_value}"
    formatted_strings.append(formatted_string)

# Print the formatted strings
for string in formatted_strings:
    print(string)