import os
from openai import AzureOpenAI
from tqdm import tqdm  # Import tqdm for the progress bar

def read_file_to_variable(file_path):
    # Get the size of the file for tqdm progress bar
    file_size = os.path.getsize(file_path)
    
    # Initialize tqdm with the file size
    # with tqdm(total=file_size, unit='B', unit_scale=True, desc=f"Reading file {file_path}") as pbar:
        # Read the content from the file
    with open(file_path, "r") as file:
       file_content = file.read()
            # pbar.update(file_size)  # Update the progress bar to show completion

    return file_content

# Example usage with six different file paths
file_paths = [
    "/media/windows/Common/_PROJECTS/new_qaribu/a_company_profile.txt",
    "/media/windows/Common/_PROJECTS/new_qaribu/b_job_descriptions.txt",
    "/media/windows/Common/_PROJECTS/new_qaribu/c_meeting_notes.txt",
    "/media/windows/Common/_PROJECTS/new_qaribu/d_project_docs.txt",
    "/media/windows/Common/_PROJECTS/new_qaribu/e_SOPs.txt",
    "/media/windows/Common/_PROJECTS/new_qaribu/f_team_structure.txt",
    
]

# Read content from each file and store in variables

a_profile, b_job_descriptions, c_meeting_notes, d_project_docs, e_SOPs, f_team_structure = [
    read_file_to_variable(file_path) for file_path in file_paths]

client = AzureOpenAI(
    # api_key="b8e6ac2cfda244dd848a823511255a0b",
    # azure_endpoint="https://hackathonservice.openai.azure.com/",
    # api_version="2023-05-15"
)

# Generate responses for each file and store in variables


# ---- INTRODUCE THE COMPANY ----

a_profile_response = client.chat.completions.create(
    model="gpt-35-turbo",
    messages=[
        {"role": "system", "content": f"You are a very friendly chatbot that's responsible for onboarding a new hire to the company described here {a_profile}. You read the text, extract meaningful info, and send clear, styled output using emojis, colours, bold and italic text. "},
        {"role": "user", "content": f"Tell me about the company"},
        {"role": "assistant", "content": "Introduce yourself as Qaribu. Welcome the user to the company, with emojis. Give a summary of the company by sending a single sentence, then a paragraph with the necessary details."},
    ]
).choices[0].message.content
print(a_profile_response)

# ----- JOB DESCRIPTIONS ----
# Describe the job descriptions and roles of the NewHire

b_job_descriptions_response = client.chat.completions.create(
    model="gpt-35-turbo",
    messages=[
        {"role": "system", "content": f"You are a very friendly chatbot that's responsible for onboarding a new hire to the company described here {b_job_descriptions}. You read the text, extract meaningful info, and send clear, styled output using emojis, colours, bold and italic text. "},
        {"role": "user", "content": f"Summarize the job descriptions for me"},
        {"role": "assistant", "content": "Send the summary of the job descriptions with a single sentence, then a paragraph with the necessary details."},
        {"role": "user", "content": f"I'll be filling the role of a Data Scientist, what does that entail?"},
        {"role": "assistant", "content": "Send the summary of the job descriptions for the role of a Data Scientist, with emojis. Use lists to be clearer. Use emojis."},

    ]
).choices[0].message.content

# Specifier, user request
# List of Job Roles
def select_role():
    # List of roles
    roles = ['Project Manager', 'Lead AI Specialist', 'AI Specialist', 'Software Developer', 'Quality Assurance Tester', 'Data Scientist']

    # Display the list of roles with corresponding letters
    for i, role in enumerate(roles, start=1):
        print(f"{chr(ord('a') + i - 1)}. {role}")

    # Prompt the user to select a role
    selected_letter = input("Which role are you interested in? Select a letter from a to f: ").lower()

    # Validate the user input
    if selected_letter.isalpha() and 'a' <= selected_letter <= 'f':
        selected_index = ord(selected_letter) - ord('a')
        selected_role = roles[selected_index]
        print(f"You selected: {selected_role}")
        return selected_role
    else:
        print("Invalid input. Please enter a letter from a to f.")
        return None

# Call the function and get the selected role
selected_role = select_role()

# Check if a role was selected
if selected_role is not None:
    # Do something with the selected role, if needed
    # For now, just print it
    print(f"Processing for role: {selected_role}")

user_request = client.chat.completions.create(
    model="gpt-35-turbo",
    messages=[
        {"role": "system", "content": f"You are a very friendly chatbot that's responsible for onboarding a new hire to the company described here {b_job_descriptions}. You read the text, extract meaningful info, and send clear, styled output using emojis, colours, bold and italic text. "},
        {"role": "user", "content": f"I am interested in being a  {selected_role} for your company. What does that entail?"},
        {"role": "assistant", "content": "Send the summary of the job descriptions with a single sentence, then a list with details. Use friendly language, bullet every emoji and show excitement. Be as gentle as possible. Only answer the question asked."},
        # {"role": "user", "content": f"I'll be filling the role of a Data Scientist, what does that entail?"},
        # {"role": "assistant", "content": "Send the summary of the job descriptions for the role of a Data Scientist, with emojis. Use lists to be clearer"},

    ]
).choices[0].message.content

print("\n Now, let's hear from you, and the role you're most excited about:")

print(user_request)

# ----- TEAM STRUCTURE ----
# Meet the team and explain the team structure

f_team_structure_response = client.chat.completions.create(
    model="gpt-35-turbo",
    messages=[
        {"role": "system", "content": f"You are a very friendly chatbot that's responsible for onboarding a new hire to the company and introducing it's team members. You read the {f_team_structure}, extract meaningful info, and send clear, styled output using emojis, colours, bold and italic text. "},
        {"role": "user", "content": f"I'd like to meet the team. Tell me about them, from {f_team_structure}"},
        {"role": "assistant", "content": "Send a list of the team members with emojis as bullets. Be friendly, and after each member, say their role, something nice about them, and what they like to do."},
    ]
).choices[0].message.content

print("\nHere's a little about our team:")
print(f_team_structure_response)

'''
def team_info():
    # Prompt the user to input "ask about our team"
    user_input = input("Type 'ask about our team': ").lower()

    # Check if the user input matches the expected string
    if user_input == 'ask about our team':
        team_member_info = input("What would you like to know about our team member? ")
        return team_member_info
    else:
        print("Invalid input. Please type 'ask about our team'.")
        return None

# Call the function and get the team member information
team_member_info = team_info()

# Check if team member information is provided
if team_member_info is not None:
    print(f"Team member information: {team_member_info}")
    # You can use the team_member_info variable as needed in your application
'''

# ----- MEETING NOTES ----
# Describe the meeting notes and to explain the culture of the company

c_meeting_notes_response = client.chat.completions.create(
    model="gpt-35-turbo",
    messages=[
        {"role": "system", "content": f"You are a very friendly chatbot that onboards a new hire to the company, and you've been sitting in on meeting. Let the new hire know what the meetings look like."},
        {"role": "user", "content": f"I'd like to know what a meeting looks like. From {c_meeting_notes}, what's the company culture like?"},
        {"role": "assistant", "content": f"Use {c_meeting_notes} to describe the company culture. Use emojis, bold and italic text. Describe what a day-to-day is like and set expectaions for behaviour."},
    ]
).choices[0].message.content

print("Here's a little about our company culture:")
print(c_meeting_notes_response)

def funniest_guy():
    # List of employee names
    employee_names = ['Azibo','Abena','Kwasi','Chike','Yewande']

    # Display the prompt and list of employee names
    print("Who do you think is the funniest?")
    for i, name in enumerate(employee_names, start=1):
        print(f"{i}. {name}")

    # Prompt the user to select the funniest person
    user_choice = input("Enter the number corresponding to your choice: ")

    # Validate the user input
    if user_choice.isdigit() and 1 <= int(user_choice) <= len(employee_names):
        selected_name = employee_names[int(user_choice) - 1]
        print(f"You selected: {selected_name}")
        return selected_name
    else:
        print("Invalid input. Please enter a number corresponding to your choice.")
        return None

# Call the function and get the selected funniest guy
funniest_employee = funniest_guy()

# Check if a funniest employee is selected
if funniest_employee is not None:
    print(f"The funniest guy is: {funniest_employee}")
    # You can use the funniest_employee variable as needed in your application

funniest_employee_response = client.chat.completions.create(
    model="gpt-35-turbo",
    messages=[
        {"role": "system", "content": f"You are a very friendly chatbot that onboards a new hire to the company, and you've been sitting in on meeting. Let the new hire know what the meetings look like."},
        {"role": "user", "content": f"Who's the funniest guy in the company?"},
        {"role": "assistant", "content": f"Tell the user that {funniest_employee} is actually really funny and one joke they said. Add emojis and laugh alot"},
    ]
).choices[0].message.content

print(funniest_employee_response)



# ----- PROJECT DOCS ----
# Describe the project to the user and explain the project

d_project_docs_response = client.chat.completions.create(
    model="gpt-35-turbo",
    messages=[
        {"role": "system", "content": f"You are an onboarding chatboat that's very friendly and methodical. You read {d_project_docs} and summarise the current project"},
        {"role": "user", "content": f"Read this file and summarize it {d_project_docs}"},
        {"role": "assistant", "content": "Send the summary for d_project_docs_response. Use emojis, bullet with emojis and be friendly"},
    ]
).choices[0].message.content

print("\nHere's a little about our current project:")
print(d_project_docs_response)

# ----- SOPs ----
# Describe the standard methods of procedure for the company

e_SOPs_response = client.chat.completions.create(
    model="gpt-35-turbo",
    messages=[
        {"role": "system", "content": "You take text files and give a one paragraph summary. Use plenty of emojis"},
        {"role": "user", "content": f"Read this file and summarize it {e_SOPs}"},
        {"role": "assistant", "content": "Send the summary for e_SOPs_response"},
    ]
).choices[0].message.content

print("\nHere's a little about our SOPs:")
print(e_SOPs_response)




# ----- RECOMMENDATIONS ----
# recommend learning paths and resources for the new hire
learning_paths_response = client.chat.completions.create(
    model="gpt-35-turbo",
    messages=[
        {"role": "system", "content": "You are nice chatbot that recommends learning paths and resources for the new hire"},
        {"role": "user", "content": f"I said i'd like to be a {selected_role}"},
        {"role": "assistant", "content": f"Send a summary of what to learn from {b_job_descriptions_response} and {d_project_docs_response}. Use emojis, bullet with emojis and be friendly"},
    ]
).choices[0].message.content

print("\nHere's a little about our current project:")
print(learning_paths_response)

print("\nHere's a few resources to get you started: ")
# List of titles and links
titles_and_links = [
"https://youtu.be/rck3MnC7OXA?si=tA2uFfV89EmhZMN8 Project Manager",
"https://youtu.be/5NgNicANyqM?si=dn2oU_h-5S75W-P5 - AI specialist",
"https://youtu.be/nu_pCVPKzTk?si=V-td48xdDrxDXK_E - Software developer",
"https://youtu.be/LZb46p8hPEg?si=Gpfevxkf6ZSAtp3Y - QA tester",
"https://youtu.be/fis26HvvDII?si=tJpw3oUIQNtih_oQ - Lead Mobile Application Developer",
"https://youtu.be/c9Wg6Cb_YlU?si=D07_Xy5X-0MQsSQT - UI/UX Designer",
"https://youtu.be/DerVeJt0OmI?si=o9tMiFt4CRa7Qnpf - Quality Assurance Tester",
"https://youtu.be/i_LwzRVP7bg?si=kQh8KTWL7NabAozB - Machine Learning Engineer",
"https://youtu.be/TPMlZxRRaBQ?si=leVeCZxcnPjJ4IBt - Tableau Developer",
"https://www.youtube.com/live/EsDFiZPljYo?si=quvVZXXCzFHEUfBG - Data Analysts",
"https://youtu.be/jcTj6FgWOpo?si=gIhpAxB4rBcZFV3m - Lead Data Analyst",
"https://youtu.be/4Z9KEBexzcM?si=3k3x6Yz4Qq6LkZ8_ - Data Scientist",]

# Print the list of strings
for item in titles_and_links:
    print(item)



# ----- PRINT RESPONSES ----
# Print or use the generated responses as needed

# print(b_job_descriptions_response)
# print(c_meeting_notes_response)
# print(d_project_docs_response)
# print(e_SOPs_response)
# print(f_team_structure_response)
