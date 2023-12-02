import requests
import gspread
from google.oauth2.service_account import Credentials
import openai
import csv
from bs4 import BeautifulSoup
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# Set up API keys
google_maps_api_key = "YOUR_GOOGLE_MAPS_API_KEY"
openai.api_key = "YOUR_OPENAI_API_KEY"
email_service_api_key = "YOUR_EMAIL_SERVICE_API_KEY"  # Placeholder for email service API key

# Step 1: Web Scraping for Lead Generation
def scrape_google_maps(location, business_type):
    url = f"https://www.google.com/maps/search/{business_type}+in+{location}"
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')

    leads = []
    for result in soup.find_all("div", class_="section-result-content"):
        name = result.find("h3", class_="section-result-title").text
        address = result.find("span", class_="section-result-location").text
        phone = result.find("span", class_="section-result-phone-number").text
        email = ""  # You may need to implement email scraping separately

        lead = {
            "Name": name,
            "Address": address,
            "Phone": phone,
            "Email": email
        }
        leads.append(lead)

    return leads

# Step 2: Write Leads to CSV
def write_leads_to_csv(leads):
    with open('leads.csv', mode='w', newline='', encoding='utf-8') as file:
        fieldnames = ['Name', 'Address', 'Phone', 'Email']
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for lead in leads:
            writer.writerow(lead)

# Step 3: Google Sheets Integration
def authenticate_google_sheets():
    scope = ["https://www.googleapis.com/auth/spreadsheets", "https://www.googleapis.com/auth/drive.file"]
    creds = Credentials.from_authorized_user_file('credentials.json', scope)
    client = gspread.authorize(creds)
    return client

def append_to_google_sheets(leads):
    client = authenticate_google_sheets()
    sheet = client.open("Leads").get_worksheet(0)
    sheet.insert_rows([list(lead.values()) for lead in leads], 2)


# Step 4: Generate Personalized Email Content
def generate_personalized_email_content(lead):
    name = lead["Name"]
    business_name = lead["Business Name"]  # Add this field if not present in leads
    location = lead["Location"]  # Add this field if not present in leads

    prompt = f"Subject: Personalized Subject for {name}\n\nBody: Dear {name},\n\nI hope this message finds you well. I came across your business, {business_name}, in {location} and was impressed by what you offer. Your dedication to {specific_service_or_aspect} caught my attention.\n\nI believe there could be a great opportunity for collaboration. Let's discuss further.\n\nBest regards,\n[Your Name]"
    
    message = openai.Completion.create(
        engine="davinci", prompt=prompt, max_tokens=150, n=1, stop=["\n"]
    ).choices[0].text.strip()

    return message

# Step 5: Email Service Integration
def setup_email_service():
    # Initialize your SMTP server and login credentials
    smtp_server = "smtp.example.com"  # Replace with your SMTP server
    port = 587  # Replace with your SMTP port (usually 587 for TLS)
    username = "your_email@example.com"  # Replace with your email
    password = "your_email_password"  # Replace with your email password

    # Establish a connection to the SMTP server
    server = smtplib.SMTP(smtp_server, port)
    server.starttls()  # Use TLS for secure connection
    server.login(username, password)

    return server

def send_email(to_email, subject, body):
    # Create a MIMEText object to represent the email
    msg = MIMEMultipart()
    msg['From'] = "your_email@example.com"  # Replace with your email
    msg['To'] = to_email
    msg['Subject'] = subject

    msg.attach(MIMEText(body, 'plain'))

    # Send the email
    server.sendmail(msg['From'], msg['To'], msg.as_string())

def close_email_service(server):
    # Close the connection to the SMTP server
    server.quit()



# Main Function
def main():
    # Step 1: Scrape leads from Google Maps
    location = "New York"
    business_type = "Restaurant"
    leads = scrape_google_maps(location, business_type)

    # Step 2: Write leads to CSV
    write_leads_to_csv(leads)

    # Step 3: Append leads to Google Sheets
    append_to_google_sheets(leads)

    # Step 4: Generate personalized email content and send emails\
    server = setup_email_service()

    for lead in leads:
        email_content = generate_personalized_email_content(lead)
        name = lead["Name"]
        email = "example@email.com"  # Replace with the lead's actual email

        # Send email using your preferred email service API
        # (You will need to implement this part)
        send_email(email, f"Personalized Subject for {name}", email_content)

        print(f"Email sent to {name} at {email}")
    
    close_email_service(server)

    

if __name__ == "__main__":
    main()
