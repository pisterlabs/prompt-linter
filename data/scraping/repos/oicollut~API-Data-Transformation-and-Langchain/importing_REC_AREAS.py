import os
import json
import openai
openai.api_key = os.getenv("OPENAI_API_KEY")
from all_funcs import write_to_google_sheet


with open("RIDBFullExport_V1_JSON/RecAreas_API_v1.json") as area_data_file:
    area_data_dict = json.load(area_data_file)
    print(f"File opened, dict assembled. Dict is {len(area_data_dict['RECDATA'])} areas long.")
    areas_dict_list = area_data_dict['RECDATA']
    
    for dict in areas_dict_list[0:500]:
        description = dict.get("RecAreaDescription", "")
        name = dict.get("RecAreaName", "")

        if len(description) < 300:
            print(f"Skipping {name} due to short description.")
            continue
        if "recreation area" in name.lower() or "recreation area" in description[:50].lower():
            #completion = openai.ChatCompletion.create(
            #model="gpt-4",
            #messages=[
                ##{"role": "system", "content": "You are a writing and editing assistant."},
                #{"role": "user", "content": f"Do a rewrite of this piece of text. Use simple language. Exclude all links. Data: {description}"}
            #]
            #)
            #print(completion.choices[0].message)
            #gpt4_text = completion.choices[0].message

            area_dict = {}

            area_dict["Name"] = dict.get("RecAreaName", "")
            area_dict["ID"] = dict.get("RecAreaID", '')
            area_dict["Description"] = description = dict.get("RecAreaDescription", "") #gpt4_text.content
            area_dict["Edited description"] = ""
            area_dict["Camping"] = ""
            area_dict["Permits"] = ""
            area_dict["Fees"] = dict.get("RecAreaFeeDescription", "")
            area_dict["Campfire"] = ""
            area_dict["BearCanisters"] = ""
            area_dict["PackAnimalsPets"] = ""
            area_dict["Parking"] = ""
            area_dict["Other"] = ""
            area_dict["AnimalsPlants"] = ""
            area_dict["GeologyHistory"] = ""
            area_dict["RisksHazards"] = ""
            area_dict["GearSkills"] = ""
            area_dict["SeasonsWeather"] = ""
            area_dict["WaterSources"] = ""
            area_dict["Address"] = f"Phone: {dict.get('RecAreaPhone', '')}, Email: {dict.get('RecAreaEmail', '')}"
            area_dict["Resources"] = ""
            area_dict["Temporary"] = ""
            area_dict["Closures"] = ""
            area_dict["RoadClosure"] = ""
            longitude = str(dict.get('RecAreaLongitude', ''))
            latitude = str(dict.get('RecAreaLatitude', ''))
            area_dict["Coordinates"] = longitude + ', ' + latitude



            worksheet = write_to_google_sheet(SHEET_ID = '1ekYANpWECKDWGmijUAw59ycT3ni1_P7AmuN9wLy1zmE', SHEET_NAME = "Recreation Areas from Rec gov")

            data_list = list(area_dict.values())
            worksheet.append_row(data_list)
            print("Writing successful! Check Google Sheet.")
        else: 
            print(f"Skipping {name} as it doesn't contain 'recreation area'")
            continue

        