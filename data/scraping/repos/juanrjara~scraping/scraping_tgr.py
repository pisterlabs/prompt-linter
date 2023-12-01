'''
This script is used to get the owners of a property in a specific comune.
It uses the TGR website to get the owners of a property by giving the rol 
and subrol of the property. It obtains a pdf file with the owner information
and then it extracts the owner name from the pdf file. It uses the 2captcha
service to solve the reCaptcha in the TGR website, so you need to have a
2captcha service key and a google site key, obtained from the TGR website to
use this script. The script also uses the PyPDF2 library to extract the text.
'''

import requests
import time
from bs4 import BeautifulSoup
import base64
from PyPDF2 import PdfReader
import re
from openai import OpenAI 

# A good practice is to set all API keys as environment variables in your system.
# Only for testing purposes, you can set the API keys here:

# Set the OpenAI API key. Note: you need to have an OpenAI account and an API key
openai_apikey = '@ltS@dm@n' # REPLACE OpenAI API key

# 2Captcha service for solving reCaptcha
service_key = 'aaaaaaaaaaa' # 2captcha service key 
google_site_key = 'bbbbbbbbbb' # Get this key from the TGR website k parameter of reCaptcha

def write_email_to_property_owner(owner_role,owner_name):
    # This function uses the OpenAI API to generate an email to the property owner
    # It uses the owner name and the owner role to generate the email. It detects
    # if the owner is a male/female person or a company and generates the email accordingly.
    # It also uses the owner role to generate the email.
    # It returns the generated email.

    client = OpenAI(api_key=openai_apikey)
    completion = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "system", "content": "You are a realtor specialized in properties."},
        {"role": "user", "content": f"Detect the sex of the property owner: {owner_name} and compose an email of maximum 100 words that tell the owner to check if he have any debt in the TGR website. The email should be addressed to the owner and should include the owner name: {owner_name} and the owner role: {owner_role}. The email should be written in a formal way."},
    ]
    )

    print(completion.choices[0].message)

#Set region code
codigo_region = 13
#Set comuna code (82 = Pudahuel)
codigo_comuna = 82
#Set each role to be searched in the array
rows = [
    "345-32",
    "386-8",
    "100-1",
    "204-7",
    "798-23",
    "643-129"
]

#Set the URL of the TGR website for captcha and scraping
pageurl = "https://web.tesoreria.cl/tramites-tgr/certificado-de-deuda-de-contribuciones"
pageurl2 = "https://www.tesoreria.cl/CertDeudasRolCutAixWeb/TraerCertificadoDeudasAction.do"

flag = False #FLAG to check if the captcha was solved (tries to solve it 10 times)
while flag == False:
    url = f"http://2captcha.com/in.php?key={service_key}&method=userrecaptcha&googlekey={google_site_key}&pageurl={pageurl}" 
    resp = requests.get(url)
    if resp.text[0:2] != 'OK': 
        quit('Service error. Error code:' + resp.text) 
    captcha_id = resp.text[3:]
    fetch_url = f"http://2captcha.com/res.php?key={service_key}&action=get&id={captcha_id}"
    for i in range(1, 10):  
        time.sleep(5) # wait 5 sec.
        resp = requests.get(fetch_url)
        if resp.text[0:2] == 'OK':
            flag = True
            break 

# Check if the captcha was solved
if flag == False:
    quit('Captcha timeout exceeded')

else:
    # Iterate over the role array
    new_rows = []
    for row in rows:
        # Split row to get rol and subrol as required by the TGR website
        rol = int(row.split("-")[0])
        subrol = int(row.split("-")[1])
        new_id = f"{rol}-{subrol}"
        
        # Post request to get the cookie JSESSIONID
        response = requests.post(pageurl2)

        # Gets the cookies from the response
        jsessionid_cookie = response.cookies.get('JSESSIONID')
        ts_cookie = response.cookies.get('TS017bac78')

        # Create a session object
        session = requests.Session()

        # Add cookies to the session
        session.cookies.set('JSESSIONID', jsessionid_cookie)
        session.cookies.set('TS017bac78', ts_cookie)

        # Define the headers
        headers = {
            'Content-Type': 'application/x-www-form-urlencoded',
            'User-Agent': 'PostmanRuntime/7.29.0',
            'Accept': '*/*',
            'Accept-Encoding': 'gzip, deflate, br'
        }

        # Define the url to get the PDF
        pageurl3 = f"{pageurl2}?region={codigo_region}&comuna={codigo_comuna}&rol={rol}&subRol={subrol}&g-recaptcha-response=&g-recaptcha-response={resp.text[3:]}"

        try:
            # Request the PDF
            response = session.post(url=pageurl3, headers=headers)
            response.raise_for_status()
            
            # parse the response using BeautifulSoup
            soup = BeautifulSoup(response.text, 'html.parser')

            # find the <iframe> with the PDF in base64
            iframe = soup.find('iframe', {'id': 'multiArPdf'})

            # Check if the <iframe> was found
            if iframe:
                # obtain <iframe> src attribute
                pdf_data = iframe['src'].replace('data:application/pdf;base64,', '')

                # decode the base64 data
                pdf_data = base64.b64decode(pdf_data)

                # save the PDF file in the current directory
                with open(f'TGR_{rol}_{subrol}_{codigo_comuna}.pdf', 'wb') as pdf_file:
                    pdf_file.write(pdf_data)

                # open the PDF file and extract the text
                with open(f'TGR_{rol}_{subrol}_{codigo_comuna}.pdf', 'rb') as pdf_file:
                    pdf_reader = PdfReader(pdf_file)
                    text = ''
                    for page in pdf_reader.pages:
                        text += page.extract_text()

                    # Define the pattern to find the name
                    nombre_pattern = r"Certificado de Deuda\n(.*?)\s+NOMBRE"

                    # Search the pattern in the text
                    match = re.search(nombre_pattern, text, re.DOTALL)

                    # Check if the name was found
                    if match:
                        nombre = match.group(1)
                        nombre = nombre.strip()
                        print(f"Rol: {new_id};Nombre: {nombre}")
                        new_row = f"{new_id}|{nombre}"
                        new_rows.append(new_row)
                        # Save the rol and name in a csv file for future use
                        with open(f"propietarios_{codigo_comuna}.csv", mode="a", encoding="utf-8") as file:
                            file.write(new_row + "\n")
                        
                        # Generate an email to the property owner
                        write_email_to_property_owner(new_id,nombre)
                    else:
                        print("Nombre no encontrado en el texto.")
            else:
                print("No se encontró el <iframe> con el PDF en base64 en la página.")

        except requests.exceptions.HTTPError as http_err:
            print(f"HTTP error occurred: {http_err}")
        except requests.exceptions.RequestException as req_err:
            print(f"Request error occurred: {req_err}")
        except Exception as err:
            print(f"An error occurred: {err}")
