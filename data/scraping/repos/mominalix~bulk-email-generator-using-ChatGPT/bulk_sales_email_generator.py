import csv
import requests
from bs4 import BeautifulSoup
import openai

# Define your OpenAI API key here ---------------------------------------- Customization required ----------------------------------------
openai.api_key =  "YOUR_API_KEY"


# Function to scrape website data and clean tags
def scrape(url):
    try:
        if url:
            response = requests.get(url)
            if response.status_code == 200:
                soup = BeautifulSoup(response.text, "lxml")
                text = []
                # <p> tags
                p_tags = []
                max_tags = 20

                names = soup.find_all("p")
                for i, tag in enumerate(names):
                    if i >= max_tags:
                        break  # Stop after extracting the first 20 tags
                    if tag.text != '':
                        name = tag.text
                        p_tags.append(name)
                text += [str(item) for item in p_tags]

                # Add additional cleaning process to the scraped text as needed


                # Return the cleaned text
                return " ".join(text)
        return ""
    except Exception as e:
        print(f"An error occurred while scraping {url}: {str(e)}")
        return ""

# Function to generate a customized email using OpenAI
def generate_customized_email(client_name, client_website, scraped_data, sender_name, sender_product_detail, sender_product_name):
    try:
        # Compose a prompt for ChatGPT
        prompt = f'''##Instruction:##
        Write a customized email to {client_name} based on the following information:
        - Client's website: {client_website}
        - Client's website data: {scraped_data}
        - Sender's product: {sender_product_detail}
        - Sender's product name: {sender_product_name}
        - Sender's name: {sender_name}
        make sure the email is from sender's product, personalized and relevant to the client's business.'''
        
        # Call the OpenAI API to generate the customized email
        response = openai.Completion.create(
            engine="text-davinci-003",
            prompt=prompt,
            max_tokens=300,  # Adjust the max_tokens as needed for your email length
            stop=None,
            temperature=0.7,
        )

        # Extract the generated email text from the API response
        generated_email = response.choices[0].text
        print(f"Generated email for {client_name}: {generated_email}")
        return generated_email
    except Exception as e:
        print(f"An error occurred while generating the email for {client_name}: {str(e)}")
        return ""

# Input and output file paths
input_csv_file = 'client_data.csv'
output_csv_file = 'customized_emails.csv'

# Read client data from the input CSV file
with open(input_csv_file, 'r') as csv_file:
    reader = csv.DictReader(csv_file)
    rows = list(reader)

# Define sender information ---------------------------------------- Customization required ----------------------------------------
sender_name = "John Wick"
sender_product_detail = "We create digital marketing campaigns that drive traffic, convert visitors into leads and sales, and scale your business fast."
sender_product_name = "Digital Alpha"

# Create or open the output CSV file for writing
with open(output_csv_file, 'w', newline='') as csv_output_file:
    fieldnames = ['Name', 'Email Address', 'Customized Email']
    writer = csv.DictWriter(csv_output_file, fieldnames=fieldnames)
    writer.writeheader()

    # Process each client's data and generate customized emails
    for row in rows:
        client_name = row['Name']
        client_email = row['Email Address']

        # Extract website URL from the email address (you may need to modify this logic)
        client_website = client_email.split('@')[1]

        # Scrape website data and clean tags
        scraped_data = scrape("http://" + client_website)  # You may need to modify the URL format

        # Generate a customized email
        customized_email = generate_customized_email(client_name, client_website, scraped_data, sender_name, sender_product_detail, sender_product_name)

        # Write the data to the output CSV file
        writer.writerow({'Name': client_name, 'Email Address': client_email, 'Customized Email': customized_email})

print(f"Customized emails have been generated and saved to {output_csv_file}.")
