import openai
import tkinter as tk
from tkinter import *
from tkinter import ttk

# Set your OpenAI API key here
api_key = "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"

# Set the API key
openai.api_key = api_key

def generate_connection_message(user_company,user, prospect, company,about , industry, position, location, common_interest, achievement, mutual_connections):
    prompt = f"Develop a short, concise introductory message from {user} of {user_company}, a local MSP company specializing in local small to medium sized businesses in {location}. The prospctful client is  {prospect} who is {position} at {company} in the {industry} industry located in  {location}. Their about me section includes{about}. Try to draw on our common iterasts in {common_interest} ,recent news or achievements for this person are {achievement} and we are mutual connections with {mutual_connections} Let's connect and explore potential synergies."

    response = openai.Completion.create(
        engine="text-davinci-002",
        prompt=prompt,
        max_tokens=500,
        n=1,
        stop=None,
        temperature=0.7
    )

    return response.choices[0].text

def generate_connection_message_gui():
    user_company = user_company_entry.get()
    user = user_entry.get()
    prospect = prospect_entry.get()
    company = company_entry.get()
    about = about_entry.get()
    industry = industry_entry.get()
    position = position_entry.get()
    location = location_entry.get()
    common_interest = common_interest_entry.get()
    achievement = achievement_entry.get()
    mutual_connections = mutual_connections_entry.get()

    # Generate the connection message
    connection_message = generate_connection_message(user_company,user,prospect, company,about , industry, position, location, common_interest, achievement, mutual_connections)

    # Display the generated connection message
    result_text.config(state=tk.NORMAL)
    result_text.delete(1.0, tk.END)
    result_text.insert(tk.END, "Generated Connection Message:\n" + connection_message)
    result_text.config(state=tk.DISABLED)

# Create the main GUI window
root = tk.Tk()
root.title("LinkedIn Connection Message Generator")

# Create input fields and labels

user_company = tk.Label(root, text="User Company Name:")
user_company.pack()
user_company_entry = tk.Entry(root, width=20)
user_company_entry.pack()

user_name = tk.Label(root, text="User Name:")
user_name.pack()
user_entry = tk.Entry(root, width=20)
user_entry.pack()

prospect_name = tk.Label(root, text="Prospect Name:")
prospect_name.pack()
prospect_entry = tk.Entry(root, width=20)
prospect_entry.pack()

company_label = tk.Label(root, text="Company:")
company_label.pack()
company_entry = tk.Entry(root, width=20)
company_entry.pack()

about_label = tk.Label(root, text="About:")
about_label.pack()
about_entry = tk.Entry(root, width=100)
about_entry.pack()

industry = tk.Label(root, text="Industry:")
industry.pack()
industry_entry = tk.Entry(root, width=20)
industry_entry.pack()

position_label = tk.Label(root, text="Position (e.g., CEO, CFO, COO):")
position_label.pack()
position_entry = tk.Entry(root, width=20)
position_entry.pack()

location_label = tk.Label(root, text="Location (e.g., Denver):")
location_label.pack()
location_entry = tk.Entry(root, width=20)
location_entry.pack()

common_interest_label = tk.Label(root, text="Common Interest:")
common_interest_label.pack()
common_interest_entry = tk.Entry(root, width=20)
common_interest_entry.pack()

achievement_label = tk.Label(root, text="Recent Achievement or Accomplishment:")
achievement_label.pack()
achievement_entry = tk.Entry(root, width=20)
achievement_entry.pack()

mutual_connections_label = tk.Label(root, text="Mutual Connections (if any):")
mutual_connections_label.pack()
mutual_connections_entry = tk.Entry(root, width=20)
mutual_connections_entry.pack()

generate_button = tk.Button(root, text="Generate Connection Message", command=generate_connection_message_gui)
generate_button.pack()

# Create a text widget to display the generated connection message
result_text = tk.Text(root, wrap=tk.WORD, width=100, height=40, state=tk.DISABLED)
result_text.pack()

# Start the GUI main loop
root.mainloop()
