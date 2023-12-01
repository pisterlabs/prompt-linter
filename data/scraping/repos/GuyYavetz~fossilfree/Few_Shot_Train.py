import gspread
from oauth2client.service_account import ServiceAccountCredentials as Credentials
import openai
import time
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# Set up Google Sheets API credentials
scope = ['https://www.googleapis.com/auth/spreadsheets',
         "https://www.googleapis.com/auth/drive.file", "https://www.googleapis.com/auth/drive"]
creds = Credentials.from_json_keyfile_name('client_s.json', scope)
client = gspread.authorize(creds)
sheet = client.open('#USE YOUR SHEET NAME HERE')
worksheet = sheet.get_worksheet(0)


openai.api_key = '#USE YOUR API KEY HERE'

# Retrieve the company descriptions from the sheet
descriptions = worksheet.col_values(15)  # Column 'O' is the 15th column

examples = [
    ("Bonus Biogrop Ltd. is a leading manufacturer of biopesticides, biostimulants, and other organic products for the agricultural and horticultural industries. Founded in 2020, the company has quickly become one of the leading suppliers of biopesticides and biostimulants in the UK and Europe. Its products are used in crop production, greenhouses, and nurseries to help maximize crop yield and quality. Bonus Biogrop Ltd. is committed to", "No"),
    ("Abujan Ltd. is a privately owned company based in the United Arab Emirates. It was established in 2014 and is a diversified business that focuses on providing innovative solutions and services to its clients in the fields of energy, infrastructure, technology, and logistics. The company is committed to delivering high-quality services and products to its customers and partners. Abujan Ltd. has a presence in the UAE, Saudi Arabia, Oman, Kuwait, Bahrain, Qatar, and India. The company has", "Yes"),
    ("Pluri Inc is a technology company based in Silicon Valley, California. Founded in 2013, the company specializes in artificial intelligence (AI) technology and services, including natural language processing, computer vision, and robotics. Pluri Inc's products and services are used by a variety of industries, including healthcare, finance, retail, and automotive. In addition to providing AI software and services, the company also offers consulting, training, and research services.", "No")
]

for i in range(1, len(descriptions)):
    prompt = "The following are examples of company classifications:\n\n"
    for j, example in enumerate(examples):
        prompt += f"{j+1}. Company: \"{example[0]}\"\n   Association with Renewable Energy: {example[1]}\n\n"

    prompt += f"Based on the above examples, determine the association with renewable energy for the following company:\n\nCompany: \"{descriptions[i]}\"\nAssociation with Renewable Energy: "

    response = openai.Completion.create(
        model="text-davinci-003", prompt=prompt, temperature=0.6, max_tokens=60)
    result = response.choices[0].text.strip()

    if "Yes" in result:
        worksheet.update_cell(i+1, 19, 1)
    else:
        worksheet.update_cell(i+1, 19, 0)

    time.sleep(0.5)  # To prevent hitting API rate limit
