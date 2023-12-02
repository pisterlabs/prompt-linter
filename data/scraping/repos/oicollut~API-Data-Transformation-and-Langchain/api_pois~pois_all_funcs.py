import os
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
import gspread
import regex
import json
#from langchain.chains import create_extraction_chain
#from langchain.schema.output_parser import OutputParserException
#import openai
#openai.api_key = os.environ["OPENAI_API_KEY"]
#from langchain.chat_models import ChatOpenAI
#llm = ChatOpenAI(temperature=0, model="gpt-4")
#gpt-3.5-turbo-16k-0613

def initialize_google_sheet(sheet_id, sheet_name):
    SCOPES = ["https://www.googleapis.com/auth/spreadsheets", "https://www.googleapis.com/auth/drive"]

    creds = None
    if os.path.exists('token.json'):
        creds = Credentials.from_authorized_user_file('token.json', SCOPES)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file('credentials.json', SCOPES)
            creds = flow.run_local_server(port=0)
        with open('token.json', 'w') as token:
            token.write(creds.to_json())

    client = gspread.authorize(creds)
    spreadsheet = client.open_by_key(sheet_id)
    worksheet = spreadsheet.worksheet(sheet_name)
    db = worksheet.get_all_records()
    #db = worksheet.get('C4:C500')

    print("Worksheet access successful.")
    return db

def extract(content: str, schema: dict):
            try:
                return create_extraction_chain(schema=schema, llm=llm).run(content)
            except OutputParserException as e:
                print(f"Error occurred while extracting: {e}")
                # handle the exception in any other way you deem necessary
                return None  # or return a default value

def write_to_google_sheet(SHEET_ID, SHEET_NAME):
    with open('token.json', 'r') as token_file:
        token_data = json.load(token_file)
    credentials = Credentials.from_authorized_user_info(token_data)

    # Initialize gspread client
    client = gspread.authorize(credentials)

    # Open the sheet
    spreadsheet = client.open_by_key(SHEET_ID)
    worksheet = spreadsheet.worksheet(SHEET_NAME)
    return worksheet

def remove_repeating_sentences(text):
    # Split the text by period to get sentences
    sentences = regex.split(r'\.\s{1,2}|(?<=[a-zA-Z])\.(?=[a-zA-Z])', text)
    sentences = [x for x in sentences if x]

    #print(sentences)
    
    unique_sentences = []
    unique_long_sentences = set()
    
    for sentence in sentences:
        # If the sentence is over 100 characters long
        if len(sentence) > 50:
            # Check if this long sentence is unique
            if sentence not in unique_long_sentences:
                # Add the long sentence to the unique set and list
                unique_long_sentences.add(sentence)
                unique_sentences.append(sentence)
        else:
            
            unique_sentences.append(sentence)
                
    cleaned_text = ". ".join(unique_sentences)
    
    return cleaned_text

def clean_wiki(text):
    cutoff_string = "Main page Contents Current events Random article"
    if cutoff_string in text:
        cutoff_string_location = text.find(cutoff_string)
    text = text[0:cutoff_string_location]
    return text
    

def split_text_into_batches(text, batch_size=5000, delimiter='.'):
    paragraphs = text.split(delimiter)
    batches = []
    current_batch = []
    current_size = 0

    for paragraph in paragraphs:
        paragraph_len = len(paragraph)
        if current_size + paragraph_len > batch_size:
            batches.append(delimiter.join(current_batch))
            current_batch = []
            current_size = 0

        current_batch.append(paragraph)
        current_size += paragraph_len

    if current_batch:  # Don't forget the last batch
        batches.append(delimiter.join(current_batch))

    return batches

def filter_data(d, substrings_to_exclude):
        
    def is_value_acceptable(value):
        if value is None or value == '':
            return False
        if isinstance(value, list):
            return all(item and item not in substrings_to_exclude for item in value)
        return value not in substrings_to_exclude

    return {k: v for k, v in d.items() if is_value_acceptable(v)}

def merge_source_data(jsons):
        merged_result = {}
        # Merge the JSONs
        for json_key, data in jsons.items():
                for key, value in data.items():
                    # If the key exists in the merged result, append the new value (if it's unique)
                    if key in merged_result:
                        if isinstance(merged_result[key], list):
                            if value not in merged_result[key]:
                                merged_result[key].append(value)
                        else:
                            if value != merged_result[key]:
                                merged_result[key] = [merged_result[key], value]
                    # If the key doesn't exist, add it to the merged result
                    else:
                        merged_result[key] = value
        print("Source data merge successful.")
        return merged_result

def get_facilities_and_services_from_campsite_list(campsite_data):
    types_of_campsites = {}
    ada_accessible_count = {}
    host_sites = 0

    for campsite in campsite_data:
        campsite_type = campsite["CampsiteType"]
        is_accessible = campsite["CampsiteAccessible"]
        
        if campsite_type == "MANAGEMENT":
            host_sites += 1
            continue
        
        types_of_campsites[campsite_type] = types_of_campsites.get(campsite_type, 0) + 1

        if is_accessible:
            ada_accessible_count[campsite_type] = ada_accessible_count.get(campsite_type, 0) + 1

    total_campsites = len(campsite_data) - host_sites

    type_mappings = {
        "STANDARD NONELECTRIC": ["is suitable for both tents and RVs", "are suitable for both tents and RVs"],
        "RV NONELECTRIC": ["is for RVs only (no electric hookups)", "are for RVs only (no electric hookups)"],
        "TENT ONLY NONELECTRIC": ["is for tents only", "are for tents only"],
        "STANDARD ELECTRIC": ["is suitable for tents and RVs (electric hookups provided)", "are suitable for tents and RVs (electric hookups provided)"],
        "WALK TO": ["a walk-to site", "walk-to sites"],
        "EQUESTRIAN NONELECTRIC": ["is reserved for camping with horses", "are reserved for camping with horses"],
        "GROUP TENT ONLY AREA NONELECTRIC": ["a tent-only group area (no electric hookups)", "tent-only group areas (no electric hookups)"],
        "GROUP SHELTER NONELECTRIC": ["is a group shelter (no electric hookups)", "are group shelters (no electric hookups)"],
        "GROUP STANDARD NONELECTRIC": ["is a group site, suitable for both tents and RVs (no electric hookups provided)", "are group sites, suitable for both tents and RVs (no electric hookups provided)"],
        "GROUP STANDARD ELECTRIC": ["is a tent and RV group site complete with electric hookups", "are tent and RV group sites complete with electric hookups"],
        "GROUP SHELTER ELECTRIC": ["is a group shelter equipped with electric hookups", "are group shelters equipped with electric hookups"],
        "RV ELECTRIC": ["is an RV-only site equipped with electric hookups", "are RV-only sites equipped with electric hookups"],
        "GROUP STANDARD AREA NONELECTRIC": ["is a group area suitable for tents and RVs (no electric hookups provided)", "are group areas suitable for tents and RVs (no electric hookups provided)"]
    }

    type_strings = []

    for campsite_type, count in types_of_campsites.items():
        accessible_count = ada_accessible_count.get(campsite_type, 0)
        
        try:
            singular, plural = type_mappings[campsite_type]
        except KeyError:
        # If the type is not in the mapping
            new_type = campsite_type.lower()
            singular = f"is {new_type}"
            plural = f"are {new_type}"
            type_mappings[campsite_type] = (singular, plural)  # Add to the mapping

        type_string = f"{count} site" if count == 1 else f"{count} sites"
        type_string += f" {singular}" if count == 1 else f" {plural}"

        if accessible_count == 1:
            type_string += " (1 of these sites is ADA accessible)"
        elif accessible_count > 1:
            type_string += f" ({accessible_count} of these are ADA accessible)"

        type_strings.append(type_string)
    if len(type_strings) > 0:
# For only one site:
        if total_campsites == 1 and 'cabin' in type_strings[0]:
            return ""
        if total_campsites == 1 and 'shelter' in type_strings[0]:
            return f"The facility features a single group shelter, which {type_strings[0].split(' ', 2)[-1]}."
        elif total_campsites == 1:
            return f"The campground features a single campsite, which {type_strings[0].split(' ', 2)[-1]}."
        # For the number of sites equal to the number of types:
        elif len(types_of_campsites) ==1 and "shelter" not in type_strings[0] and "cabin" not in type_strings[0]:
            return f"The campground has {total_campsites} campsites, all of which are {type_strings[0].split(' ', 3)[-1]}."
        elif len(types_of_campsites) ==1 and "shelter" in type_strings[0]:
            return f"The facility features {total_campsites} sites, all which are {type_strings[0].split(' ', 3)[-1]}."
        else:
            facilities_string = f"The campground has {total_campsites} campsites, of which"
            if len(type_strings) == 1:
                facilities_string += ' ' + type_strings[0]
            elif len(type_strings) > 1:
                facilities_string += ' ' + ', '.join(type_strings[:-1]) + ' and ' + type_strings[-1]
            facilities_string += '.'
    elif total_campsites == 1:
        facilities_string = f"The campground features {total_campsites} campsite."
    else: 
        facilities_string = f"The campground features {total_campsites} campsite."
        
    return facilities_string
    

def get_campsite_attributes():
    with open("api_pois/CampsiteAttributes_API_v1.json") as attributes_data_file:
        attributes_data_dict = json.load(attributes_data_file)
        print(f"File opened, dict assembled. Dict is {len(attributes_data_dict.values())} campsites long.")
    #print(f"Iterating over dict to find target {len(campsite_IDs)} campgrounds.")
        for key, value in attributes_data_dict.items():
            recdata = key
            campsite_dict_list = value
            all_campground_dicts = {}
            for campsite_dict in campsite_dict_list:
                data_dict = {}
                if "EntityID" in campsite_dict.values():
                    data_dict["CampsiteID"] = campsite_dict["AttributeValue"]
                if "Capacity/Size Rating" in campsite_dict.values():
                    data_dict["CampsiteSize"] = campsite_dict["AttributeValue"]
                if "Checkin Time" in campsite_dict.values():
                    data_dict["CheckIn"] = campsite_dict["AttributeValue"]
                if "Checkout Time" in campsite_dict.values():
                    data_dict["CheckOut"] = campsite_dict["AttributeValue"]
                if "Max Num of People" in campsite_dict.values():
                    data_dict["MaxPeople"] = campsite_dict["AttributeValue"]
                if "Max Num of Vehicles" in campsite_dict.values():
                    data_dict["MaxVehicles"] = campsite_dict["AttributeValue"]
                if "Proximity to Water" in campsite_dict.values():
                    data_dict["Waterfront"] = campsite_dict["AttributeValue"]
                if "Campfire Allowed" in campsite_dict.values():
                    data_dict["CampfireAllowed"] = campsite_dict["AttributeValue"]
                all_campground_dicts.update(data_dict)
    return all_campground_dicts
 

def get_rules_and_regs_string(id):
    with open("/Users/Sasha/Documents/Python_Stuff/Langchain/MY_CAMPSITE_ATTRIBUES.json") as attributes_data_file:
        data_dict = json.load(attributes_data_file)
        if data_dict.get(id):
            rules_string = format_rules_and_regs(data_dict[id])
        else: rules_string = f"No campsite with id {id} in Campsite Attributes file." 
    print(rules_string)
    return rules_string
    #return data_dict

def format_rules_and_regs(campsite_attrs: dict):
    check_in = campsite_attrs.get('CheckIn', '')
    check_out = campsite_attrs.get('CheckOut', '')

    people_rules = []
    vehicle_rules = []

    #campsite_size = campsite_attrs.get('CampsiteSize', '')
    max_people = campsite_attrs.get('MaxPeople', '')
    max_vehicles = campsite_attrs.get('MaxVehicles', '')
    
    #if len(campsite_size) > 1 and len(max_people) > 0:
        #people_rules.append(f"{max_people} per {campsite_size} campsite")
        #vehicle_rules.append(f"{max_vehicles} per {campsite_size} campsite")
    if len(max_people) > 0:
        people_rules.append(f"{max_people} per campsite")
        vehicle_rules.append(f"{max_vehicles} per campsite")
        
    people_string = " and ".join(people_rules)
    vehicle_string = " and ".join(vehicle_rules)

    if len(check_in) > 1 and len(check_out) > 1:
        check_in_segment = f"Check-in time is {check_in}, check-out is at {check_out}."
    else:
        check_in_segment = ""
    if len(max_people) > 0 and (people_string != "0"):
        people_segment = f"The maximum number of people is {people_string}."
    else:
        people_segment = ""
    if len(max_vehicles) > 0 and vehicle_string != "0":
        vehicle_segment = f"The maximum number of vehicles is {vehicle_string}."
    else: vehicle_segment = ""
    
    rules_string = check_in_segment + "\n" + people_segment + "\n" + vehicle_segment + "\n"

    return rules_string

def get_area_dict(file_path):
    with open(file_path, 'r') as file:
        data_dict = json.load(file)
    
    return data_dict

def get_area_name(data_dict, area_id):
    import json

    # Check if 'RECDATA' is in the dictionary
    if 'RECDATA' in data_dict:
        # Loop through each item in 'RECDATA'
        for item in data_dict['RECDATA']:
            # Check if the ID matches
            if item.get('RecAreaID') == area_id:
                # Return the corresponding name
                return item.get('RecAreaName')
    return None


def remove_tags(text):
    import re
    pattern = r'<[^>]+>'

    clean_text = re.sub(pattern, '', text)

    return clean_text

"""attributes_data_dict = {"RECDATA":[
    {"AttributeID":9,"AttributeName":"Campfire Allowed","AttributeValue":"Yes","EntityID":"1","EntityType":"Campsite"},
    {"AttributeID":10,"AttributeName":"Capacity/Size Rating","AttributeValue":"Single","EntityID":"1","EntityType":"Campsite"},
    {"AttributeID":11,"AttributeName":"Checkin Time","AttributeValue":"12:00 PM","EntityID":"1","EntityType":"Campsite"},
    {"AttributeID":12,"AttributeName":"Checkout Time","AttributeValue":"12:00 PM","EntityID":"1","EntityType":"Campsite"},
    {"AttributeID":23,"AttributeName":"Driveway Entry","AttributeValue":"Parallel","EntityID":"1","EntityType":"Campsite"},
    {"AttributeID":24,"AttributeName":"Driveway Grade","AttributeValue":"Slight","EntityID":"1","EntityType":"Campsite"},
    {"AttributeID":25,"AttributeName":"Driveway Length","AttributeValue":"","EntityID":"1","EntityType":"Campsite"},
    {"AttributeID":26,"AttributeName":"Driveway Surface","AttributeValue":"Gravel","EntityID":"1","EntityType":"Campsite"},
    {"AttributeID":52,"AttributeName":"Max Num of People","AttributeValue":"6","EntityID":"1","EntityType":"Campsite"},
    {"AttributeID":53,"AttributeName":"Max Num of Vehicles","AttributeValue":"2","EntityID":"1","EntityType":"Campsite"},
    {"AttributeID":54,"AttributeName":"Max Vehicle Length","AttributeValue":"27","EntityID":"1","EntityType":"Campsite"},
    {"AttributeID":65,"AttributeName":"Pets Allowed","AttributeValue":"Yes","EntityID":"1","EntityType":"Campsite"},
    {"AttributeID":72,"AttributeName":"Proximity to Water","AttributeValue":"Riverfront","EntityID":"1","EntityType":"Campsite"},
    {"AttributeID":314,"AttributeName":"Placed on Map","AttributeValue":"1","EntityID":"1","EntityType":"Campsite"},
    {"AttributeID":10429,"AttributeName":"IS EQUIPMENT MANDATORY","AttributeValue":"true","EntityID":"1","EntityType":"Campsite"}
    ]
}"""


def get_campsite_attributes_json():
    import json
    with open("api_pois/CampsiteAttributes_API_v1.json") as attributes_data_file:
        attributes_data_dict = json.load(attributes_data_file)
        print(f"File opened, dict assembled. Dict is {len(attributes_data_dict['RECDATA'])} campsites long.")
    
    all_campground_dicts = {}
    
    for campsite_dict in attributes_data_dict['RECDATA']:
        entity_id = campsite_dict.get("EntityID")
        attribute_name = campsite_dict.get("AttributeName")
        attribute_value = campsite_dict.get("AttributeValue")
        
        if entity_id not in all_campground_dicts:
            all_campground_dicts[entity_id] = {}
            
        if attribute_name == "Capacity/Size Rating":
            all_campground_dicts[entity_id]["CampsiteSize"] = attribute_value
        elif attribute_name == "Checkin Time":
            all_campground_dicts[entity_id]["CheckIn"] = attribute_value
        elif attribute_name == "Checkout Time":
            all_campground_dicts[entity_id]["CheckOut"] = attribute_value
        elif attribute_name == "Max Num of People":
            all_campground_dicts[entity_id]["MaxPeople"] = attribute_value
        elif attribute_name == "Max Num of Vehicles":
            all_campground_dicts[entity_id]["MaxVehicles"] = attribute_value
        elif attribute_name == "Proximity to Water":
            all_campground_dicts[entity_id]["Waterfront"] = attribute_value
        elif attribute_name == "Campfire Allowed":
            all_campground_dicts[entity_id]["CampfireAllowed"] = attribute_value
    
    return all_campground_dicts


def get_segments_from_recgov_decsription(description: str):
    import re

    if '<h2>' not in description:
        return [description]

    matches = re.findall(r'<h2>(.*?)</h2>(.*?)<h2>', description, re.DOTALL)

    sections = {match[0]: match[1].strip() for match in matches}

    last_h2 = re.search(r'<h2>([^<]+)</h2>(.*)', description.split("<h2>")[-1], re.DOTALL)

    if last_h2:
        sections[last_h2.group(1)] = last_h2.group(2).strip()

    # Now, based on your headers, you can assign the values:
    main_description = sections.get('Overview', '')
    recreation = sections.get('Recreation', '')
    facilities = sections.get('Facilities', '')
    natural_features = sections.get('Natural Features', '')
    nearby_attractions = sections.get('Nearby Attractions', '')
    contact_info = sections.get('contact_info', '') + sections.get('Contact Info', '')
    rules_and_regs = sections.get('Charges &amp; Cancellations', '') + sections.get('Know Before You Go', '')

    string_list = []
    string_list.append(main_description)
    string_list.append(natural_features)
    string_list.append(recreation)
    string_list.append(facilities)
    string_list.append(nearby_attractions)
    string_list.append(contact_info)
    string_list.append(rules_and_regs)

    print("Main Description:\n", main_description)
    print("\nRecreation:\n", recreation)
    print("\nFacilities:\n", facilities)
    print("\nNatural Features:\n", natural_features)
    print("\nNearby Attractions:\n", nearby_attractions)
    print("\nContact Info:\n", contact_info)
    print("\nRules and Regulations:\n", rules_and_regs)

    return(string_list)


def linkcheck(camp_id):
    import requests
    link1 = f"https://www.recreation.gov/camping/campgrounds/{camp_id}"
    link2 = f"https://www.recreation.gov/camping/poi/{camp_id}"

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }

    # Check if the link works
    response = requests.get(link1, headers=headers)
    if response.status_code == 200 and "Recreation.gov is your source for discovering and experiencing America\'s federal recreation activities and outdoor adventures." not in response.text:
        #print(response.text)
        message = f'Reservations can be made online at <a href="{link1}">Recreation.gov</a> or by calling 1-877-444-6777.'
    else:
        response = requests.get(link2, headers=headers)
        if response.status_code == 200 and "Recreation.gov is your source for discovering and experiencing America\'s federal recreation activities and outdoor adventures." not in response.text:
            #print(response.text)
            message = f'More information is available at <a href="{link2}">Recreation.gov</a>.'
        else: message = ""


    return message

#message = linkcheck("10109092")
#print(message)

def scrape_website(url: str):
    import requests
    from bs4 import BeautifulSoup
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.3'
    }

    try:
        response = requests.get(url, headers=headers, timeout=10)
        
        # Check if the response status is not 200 (OK)
        if response.status_code != 200:
            print(f"Error: Received status code {response.status_code} when accessing the website.")
            return

        soup = BeautifulSoup(response.content, 'html.parser')
        content = soup.prettify()

        # Check if content seems too short
        if len(content) < 100:
            print("Warning: The scraped content seems unusually short.")

        # Check if there are common anti-scraping measures
        if "robot" in content or "captcha" in content:
            print("Warning: The website might have anti-scraping measures in place.")

        return content

    except requests.RequestException as e:
        print(f"Error: Unable to access the website. Error: {e}")
        return

    except Exception as e:
        print(f"Unexpected error: {e}")
        return

def extract_content(html_content: str):
     
    starting_substring = "Need to Know"
    starting_position = html_content.find(starting_substring)
    if starting_position != -1:
        print(f"'{starting_substring}' was found in the main string at position {starting_position}!")
    else:
        print(f"'{starting_substring}' was not found in the main string.")

    ending_substring = "Natural Features"

    ending_position = html_content.find(ending_substring)
    if starting_position != -1:
        print(f"'{ending_substring}' was found in the main string at position {ending_position}!")
    else:
        print(f"'{ending_substring}' was not found in the main string.")
    
    need_to_know = html_content[starting_position: ending_position]

    
    return need_to_know



def extract_wanted_rules(content: str):

    import re
    
    content = content.replace("<p>", "")
    content = content.replace("</p>", "")
    bear_info = re.search(r"<li>[^<]*[Bb]ear[^<]*<\/li>", content, re.MULTILINE | re.IGNORECASE)
    vehicle_length = re.search(r"<li>[^<]*[Vv]ehicle[^<]*<\/li>",  content, re.MULTILINE | re.IGNORECASE)
    pets_info = re.search(r"<li>[^<]*[Pp]et[^<]*<\/li>", content, re.MULTILINE | re.IGNORECASE)

    extracted_info = []

    if vehicle_length:
        vehicles = remove_tags(vehicle_length.group())
        extracted_info.append(" ".join(vehicles.split()))
    if bear_info:
        bears = remove_tags(bear_info.group())
        extracted_info.append(" ".join(bears.split()))
    if pets_info:
        pets = remove_tags(pets_info.group())
        extracted_info.append(" ".join(pets.split()))

    extracted_info = "\n".join(extracted_info)

    return extracted_info

