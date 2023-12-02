import os
import openai
openai.api_key = os.getenv("OPENAI_API_KEY")
from all_funcs import initialize_google_sheet, write_to_google_sheet
import json

#db = initialize_google_sheet(sheet_id='1I21v0eu5sAeEb0ZwirantxABsx6E8Cnr2DlZACEkgAY', sheet_name='Sheet5')

data = {'protected_area_history_broken_up_into_very_short_sentences': ["The park has a history of intense mercury mining. \nThis dates back to the Gold Rush Era. \nMercury from the park was used in gold and silver mines. \nThese mines are located in the Sierras. \nMining operations have since ceased. \nMost tunnels from mining have been sealed. \nRemnants of mining can be seen throughout the park. The Ohlone people were based in and around what is now Almaden Quicksilver County Park. \nThe region has deposits of cinnabar, used by the Ohlone for paint. \nIn the 1840s, a Mexico-based English textile firm started mining the region. \nIt was named Nueva Almaden. \nNew Almaden was the oldest and most productive mine in the US.\nBy 1865, there were 700 buildings and 1,800 residents on Mine Hill. \nLarge-scale mining stopped in 1927. The mining led to mercury pollution in the Guadalupe River and San Francisco bay. Quicksilver Mining ended in the 1970s due to environmental concerns over mercury. Santa Clara County got the first parcel of the park in 1973. \nIt was later marked as a Superfund site and a National Historic Site. \nSanta Clara County turned the mine into a park in the 1970s.  \nMine Hill was cleaned and made accessible to the public. \n\nThe park features remains of mining facilities. \nSan Cristobal Mine is a key location within the park"]}

main_description_keys = ['protected_area_name', 'protected_area_location', 'protected_area_recreational_opportunities', 'protected_area_landscape_features', 'protected_area_visitor_infrastructure']
optional_metadata_keys = ["protected_area_flora", "protected_area_fauna", "protected_area_history", "protected_area_geology", 
                          "protected_area_address", "protected_area_contacts", "protected_area_fees", "protected_area_camping_options",
                          "protected_area_parking", "protected_area_campfire_rules", "protected_area_food_storage_rules", "protected_area_dog_rules"]
""""
present_optional_keys = []
for entry in db:
    area_data = entry["data"]
    for key in optional_metadata_keys:
        if key in area_data:
            present_optional_keys.append(key)
    #print(area_data)

    key_mapping = {
    "protected_area_flora": "Plant Life",
    "protected_area_fauna": "Wildlife",
    "protected_area_history": "History",
    "protected_area_geology": "Geology",
    "protected_area_address": "Address",
    "protected_area_contacts": "Contacts",
    "protected_area_fees": "Park Fees",
    "protected_area_camping_options": "Camping Options",
    "protected_area_parking": "Parking Information",
    "protected_area_campfire_rules": "Campfire Rules",
    "protected_area_food_storage_rules": "Food Storage Rules",
    "protected_area_dog_rules": "Pet Rules"
    }

    present_optional_keys = [key_mapping.get(key, key) for key in present_optional_keys]


    print(present_optional_keys)
    present_optional_keys = "; ".join(present_optional_keys)
    """

completion = openai.ChatCompletion.create(
model="gpt-4",
messages=[
    {"role": "system", "content": "RightOnTrek is an encyclopedia for hikers in the US."},
    {"role": "user", "content": f"Write a RightOnTrek hiking encyclopedia paragraph about the history of Almaden Quicksilver County Park. Use given data. Character limit 700. Data: {data}"}
]
)
print(completion.choices[0].message)
gpt4_text = completion.choices[0].message
#worksheet = write_to_google_sheet(SHEET_ID = '1I21v0eu5sAeEb0ZwirantxABsx6E8Cnr2DlZACEkgAY', SHEET_NAME = 'Sheet5')

#next_row = len(worksheet.col_values(3)) + 1

# Append the forest_name and data_str to the sheet
#worksheet.update_cell(next_row + 3, 3, gpt4_text.content)
#print("Writing successful! Check Google Sheet.")

