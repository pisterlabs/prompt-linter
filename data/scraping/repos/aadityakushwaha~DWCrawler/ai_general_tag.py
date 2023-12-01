import mysql.connector
import openai
import time

# Connect to the SQL database
db = mysql.connector.connect(
  host="localhost",
  user="root",
  password="",
  database="Crawler"
)

# Retrieve the data from the database
cursor = db.cursor()
cursor.execute("SELECT * FROM onion_urls WHERE ai_general_tag IS NULL")
rows = cursor.fetchall()

# Set up the OpenAI API
openai.api_key = "<API-KEY>"
model_engine = "text-davinci-003"
initial = "Category are:\n1. Ransomware\n2. Botnets\n3. Darknet markets\n4. Bitcoin services\n5. Hacking groups and services\n6. Financing and fraud\n7. Illegal pornography\n8. Terrorism\n9. Social media\n10. Hoaxes and unverified content\n11. Weapons\n12. MISC"
prompt_template = "Please provide a category for this website based on the following information:\nTitle: {}\nDescription: {}\n"
prompt_template = prompt_template + initial

# Loop through each row of data
for row in rows:
    # Extract the relevant data
    id_ = row[0]
    url = row[1]
    title = row[2]
    keywords = row[3]
    description = row[4]
    content = row[5]
    
    # Generate the prompt for the OpenAI API
    prompt = prompt_template.format(title, description)
    
    try:
        # Call the OpenAI API to generate tags
        response = openai.Completion.create(
            engine=model_engine,
            prompt=prompt,
            max_tokens=50,
            n=1,
            stop=None,
            temperature=0.7,
        )
    
        # Extract the generated tags from the API response
        tags = response.choices[0].text.strip()
    
        # Update the corresponding row in the database with the generated tags
        update_query = "UPDATE onion_urls SET ai_general_tag = %s WHERE id = %s"
        cursor.execute(update_query, (tags, row[0]))
        db.commit()
        print(f"tags for {id_} and {url} : " + tags + "\n")
        
        # Sleep for 1 second to avoid rate limit errors
        time.sleep(1)
        
    except openai.error.InvalidRequestError as e:
        print("Error: String too large\n")
    except openai.error.RateLimitError as e:
        print("Rate Exceeded\n")
        # Sleep for 60 seconds to wait for the rate limit to reset
        time.sleep(60)

# Close the database connection
db.close()
