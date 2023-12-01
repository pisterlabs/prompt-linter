import openai
import sqlite3

# Set up your OpenAI API key
openai.api_key = "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"

def get_person_details(name):
    # Define the initial conversation prompt
    conversation = [
        {"role": "system", "content": "You are a helpful assistant that provides details about people."},
        {"role": "user", "content": f"What can you tell me about {name}?"}
    ]

    # Generate the response using the Chat API
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=conversation
    )

    # Extract and return the assistant's reply
    reply = response.choices[0].message['content']
    return reply

def insert_person_details(conn, name, details):
    cursor = conn.cursor()
    cursor.execute('INSERT INTO people (name, details) VALUES (?, ?)', (name, details))
    conn.commit()
    cursor.close()

def get_person_details_from_db(conn, name):
    cursor = conn.cursor()
    cursor.execute('SELECT details FROM people WHERE name = ?', (name,))
    row = cursor.fetchone()
    cursor.close()

    if row:
        return row[0]
    else:
        return None

# Connect to the SQLite database
conn = sqlite3.connect('people.db')

# Create the people table if it doesn't exist
cursor = conn.cursor()
cursor.execute('''
    CREATE TABLE IF NOT EXISTS people (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT,
        details TEXT
    )
''')
cursor.close()

# Get the name from the user
name = input("Enter the name of the person: ")

# Check if details are available in the database
details = get_person_details_from_db(conn, name)

if details is None:
    # Get the person details using the conversational agent
    details = get_person_details(name)

    # Store the details in the database
    insert_person_details(conn, name, details)
    print("Details stored in the database.")
else:
    print("Details found in the database.")

# Print the details
print(details)

# Close the database connection
conn.close()
