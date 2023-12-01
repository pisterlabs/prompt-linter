import json
import gspread
from google.oauth2.credentials import Credentials
#from google_auth_oauthlib.flow import InstalledAppFlow
from all_funcs import initialize_google_sheet, remove_repeating_sentences, clean_wiki, merge_source_data, extract, filter_data
from langchain.document_loaders import AsyncChromiumLoader
from langchain.document_transformers import BeautifulSoupTransformer
#from google.auth.transport.requests import Request
 # define Large Language Model to be used 
                                                                # and set temp (model's 'creativity') to zero

db = initialize_google_sheet(sheet_id='1ZA9WVAAhHpmf5ikwPv1W3k1CqYNGY_Zatu5MZTrjkcg', sheet_name='Sheet1') # access Google Sheet with source websites with Google's OAuth2

# Iterate over Forests (areas) and their sources in the Sheet, scraping source by source and storing scraped data in dict (jsons)
forest_count = 0
merged_jsons = {}

for entry in db:
    jsons = {}
    source_count = 0
    for source in entry['Sources'].split(","):
        print('Loading source', source)

        loader = AsyncChromiumLoader([source])
        html = loader.load() # load source link
        print("Parsing source", source)
        bs_transformer = BeautifulSoupTransformer()
        docs_transformed = bs_transformer.transform_documents(html,tags_to_extract=["p", "li", "div", "a", "span"]) # scrape pages with BS
        print("Removing clutter from parsed source text...")
        docs_transformed = docs_transformed[0].page_content[0:]
        if 'wikipedia.org' in source:
            docs_transformed = clean_wiki(docs_transformed) # store entire scraped text as string (sample below)
        cleaned_scrapings = remove_repeating_sentences(docs_transformed)[0:7000]
        print(cleaned_scrapings[:100])
        if html[0].page_content == '':
            print('error loading source')
            continue

        # docs_transformed_sample_data = "On March 1, 1872, Yellowstone became the first national park for all to enjoy the unique hydrothermal and geologic features. Within Yellowstone's 2.2 million acres, visitors have unparalleled opportunities to observe wildlife in an intact ecosystem, explore geothermal areas that contain about half the world’s active geysers, and view geologic wonders like the Grand Canyon of the Yellowstone River."
    
        # Define a schema based on which the LLM will extract data from the scraped text
        # The schema is essentially a prompt and can be defined and redefined based on our needs
        
        schema = {
                    "properties": {
                    "protected_area_name": { "type": "string" },
                    "protected_area_size": { "type": "string" },
                    "protected_area_location": { "type": "string" },
                    "protected_area_nearby_attractions": { "type": "string" },
                    "protected_area_landscape_features": { "type": "string" },
                    "protected_area_famous_places_to_visit": { "type": "string" },
                    "protected_area_recreational_opportunities": { "type": "string" },
                    "protected_area_visitor_infrastructure": { "type": "string" },
                    #"protected_area_flora": { "type": "string" },
                    #"protected_area_fauna": { "type": "string" },
                    "protected_area_history_broken_up_into_very_short_sentences": { "type": "string" },
                    #"protected_area_geology": { "type": "string" },
                    #"protected_area_address": { "type": "string" },
                    #"protected_area_contacts": { "type": "string" },
                    #"protected_area_fees": { "type": "string" },
                    #"protected_area_camping_options": { "type": "string" },
                    #"protected_area_parking": { "type": "string" },
                    #"protected_area_campfire_rules": { "type": "string" },
                    #"protected_area_food_storage_rules": { "type": "string" },
                    #"protected_area_dog_rules": { "type": "string" }
                },
                "required": ["protected_area_name", "protected_area_size", "protected_area_location"]
                }


        protected_area_data = extract(cleaned_scrapings, schema)
        if protected_area_data is None:
            # handle the error case if needed
            pass


        if not protected_area_data:
            print(f"Error: protected_area_data is empty for source: {source}")
            continue

        if (isinstance(protected_area_data, list)):
            protected_area_data = protected_area_data[0]

        print(f"Data for forest_{forest_count} from source_{source_count} extracted!")
        print("Processing data...")

        substring_to_clear = ["not mentioned", "Not mentioned", "no data", "None", "none", '']

        protected_area_data = filter_data(protected_area_data, substring_to_clear)
        print("Filtered dictionary:", protected_area_data)

        source_key_name = f"json_{source_count}" # Store each result under its source name 
      
        jsons[source_key_name] = protected_area_data

        source_count += 1

    # Now we have a dict with data for each source for the same forest (see sample above), 
    # we can merge them into a single dictionary, keeping only the unique data
    
    
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

    print(merged_result)
    
    # Store merged data under the relevant Forest Number
    forest_number = f"forest_{forest_count}"
    if forest_number not in merged_jsons.keys():
        merged_jsons[forest_number] = merge_source_data(jsons)

    forest_count +=1
    print(merged_jsons)

    print("Processing successful.")
# Access the target Sheet to write the gathered data
    break
"""""
with open('token.json', 'r') as token_file:
    token_data = json.load(token_file)
credentials = Credentials.from_authorized_user_info(token_data)

# Initialize gspread client
client = gspread.authorize(credentials)

# Open the sheet
SHEET_ID = '1I21v0eu5sAeEb0ZwirantxABsx6E8Cnr2DlZACEkgAY'
SHEET_NAME = 'Sheet1'
spreadsheet = client.open_by_key(SHEET_ID)
worksheet = spreadsheet.worksheet(SHEET_NAME)

# Write data row by row, adding the forest's official name to col A and dumping all data to col B
for forest_number, data in merged_jsons.items():
    if isinstance(data['protected_area_name'], list):
        forest_name = data['protected_area_name'][0]
    else:
        forest_name = data['protected_area_name']
# Convert the dictionary data to a string for writing to the sheet
    data_str = str(data)

    # Append the forest_name and data_str to the sheet
    worksheet.append_row([forest_name, data_str])

    print("Writing successful! Check Google Sheet.")"""