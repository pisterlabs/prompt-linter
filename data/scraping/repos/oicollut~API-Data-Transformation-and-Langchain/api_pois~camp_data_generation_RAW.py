import os
#import openai
#openai.api_key = os.getenv("OPENAI_API_KEY")
from pois_all_funcs import write_to_google_sheet, get_facilities_and_services_from_campsite_list, get_campsite_attributes, get_rules_and_regs_string, get_segments_from_recgov_decsription, linkcheck, remove_tags, scrape_website, extract_content, extract_wanted_rules, get_area_dict, get_area_name
import json
from itertools import islice
import time


with open("api_pois/CAMPGROUNDS.json") as campground_data_file:
    campground_data_dict = json.load(campground_data_file)
    print(f"File opened, dict assembled. Dict is {len(campground_data_dict)} campgrounds long.")
    #x = 250
    x = 1
    y = 300

    campground_names = [
    "BADGER FLATS GROUP",
    "UPPER BILLY CREEK CG",
    "CATAVEE",
    "COLLEGE",
    "DEER CREEK",
    "RANCHERIA",
    "MAMMOTH POOL",
    "Rock Creek (Sierra National Forest, CA)",
    "FISH CREEK (CA)",
    "DORABELLE CAMPGROUND"
]
    data_to_write = []
    #print(f"Iterating over dict to find target {len(campground_names)} campgrounds.")
    for key, value in islice(campground_data_dict.items(), x, y): 
        camp_name = key
        camp_name_value = value
        campsite_data_situation = camp_name_value[1]
        facility = camp_name_value[0]
        description_data = facility['FacilityDescription']
        facility_id = facility['FacilityID']

        description_segmented_list = get_segments_from_recgov_decsription(description_data)

        main_description = remove_tags(description_segmented_list[0]) if len(description_segmented_list) > 0 else ""
        natural_features = remove_tags(description_segmented_list[1]) if len(description_segmented_list) > 1 else ""
        recreation = remove_tags(description_segmented_list[2]) if len(description_segmented_list) > 2 else ""
        facilities = remove_tags(description_segmented_list[3]) if len(description_segmented_list) > 3 else ""
        nearby_attractions = remove_tags(description_segmented_list[4]) if len(description_segmented_list) > 4 else ""
        contact_info = remove_tags(description_segmented_list[5]) if len(description_segmented_list) > 5 else ""
        rules_and_regs_from_description = remove_tags(description_segmented_list[6]) if len(description_segmented_list) > 6 else ""

        resources = linkcheck(facility_id)

        page_to_scrape = scrape_website(f"https://www.recreation.gov/camping/campgrounds/{facility_id}?tab=info")
        if page_to_scrape:
            rules_and_regs_from_website = extract_content(page_to_scrape)
            print(rules_and_regs_from_website)
            bears_pets_vehicles_rules = extract_wanted_rules(rules_and_regs_from_website)

        parent_area_id = facility.get("ParentRecAreaID", "")
        if len(parent_area_id) > 0:
            area_data = get_area_dict("/Users/Sasha/Documents/Python_Stuff/Langchain/RIDBFullExport_V1_JSON/RecAreas_API_v1.json")
            area_name = get_area_name(area_data, parent_area_id)
        
        data_dict = {
            "POI_ID" : "",
            "Area_ID" : facility.get("ParentRecAreaID", ""),
            "Area_Name": area_name,
            "Camp_ID" : facility.get("FacilityID", ""),
            "Name" : facility.get("FacilityName", ""),
            "Accessibility" : facility.get("FacilityAdaAccess", '') if facility.get("FacilityAdaAccess") else "",
            "Pets" : "", #"Enter info if available"
            "Alert" : "",
            "MainDescription" : main_description + "\n" + recreation + "\n" + nearby_attractions,
            "HowToGetThere" : remove_tags(facility.get("FacilityDirections", "")),
            "Address" : "", #"Show directions on Google Maps"
            "ContactInformation" : f"Phone: {facility.get('FacilityPhone', '')}" if facility.get("FacilityPhone") else "",
            "Email" : f"Email: {facility.get('FacilityEmail', '')}" if facility.get("FacilityEmail") else "",
            "Resources" : resources,
            "Season" : "", #f"{data_dict['Name'].lower()} Campground | USFS" + '\n' + 
            "RulesAndRegulations" : get_rules_and_regs_string(id) + "\n" + str(bears_pets_vehicles_rules) + "\n" + rules_and_regs_from_description

        }

        #if camp_name in campground_names:
        if campsite_data_situation == "NO ADDITIONAL CAMPSITE DATA":
            print("Executing scenario where no campsite data is given.")
            
            data_dict["FacilitiesAndServices"] = facilities
            data_dict["Coordinates"] = f"{facility.get('FacilityLongitude', '')}, {facility.get('FacilityLatitude', '')}"
        
            #FOR CAMPGROUNDS WITH INDIVIDUAL CAMPSITE INFO
        else:
            print("Executing scenario where we have campsite data.")
        
            campsite_data = camp_name_value[1]

            facilities_string = get_facilities_and_services_from_campsite_list(campsite_data)
            if "ADA accessible" in facilities_string:
                data_dict["Accessibility"] = "Y"
            id = campsite_data[0]['CampsiteID']
        
            data_dict["RulesAndRegulations"] = get_rules_and_regs_string(id) + "\n" + str(bears_pets_vehicles_rules) + "\n" + remove_tags(rules_and_regs_from_description)
            data_dict["FacilitiesAndServices"] = f"{facilities_string}" + "\n" + facilities
            data_dict["Coordinates"] = f"{facility.get('FacilityLongitude', '')}, {facility.get('FacilityLatitude', '')}"

        data_list = list(data_dict.values())
        data_to_write.append(data_list)
        print(f"Processed campground {key}")

    covered_area_names = [
    "Mission Mountains Wilderness",
    "Mission Mountains Tribal Wilderness",
    "Kootenai National Forest",
    "Cabinet Mountain Wilderness",
    "Caribou-Targhee National Forest",
    "Whitefish Lake State Park",
    "Siskiyou Wilderness",
    "Smith River National Recreation Area",
    "Jedediah Smith Redwoods State Park",
    "Del Norte Coast Redwoods State Park",
    "Tolowa Dunes State Park and Lake Earl Wildlife Area",
    "Redwood National Park",
    "Humboldt Redwoods State Park",
    "Humboldt Lagoons State Park",
    "Patricks Point State Park",
    "Trinidad State Beach",
    "Richardson Grove State Park",
    "Benbow Lake State Recreation Area",
    "King Range National Conservation Area + King Range Wilderness",
    "Lava Beds National Monument",
    "Klamath National Forest",
    "Shasta-Trinity National Forest",
    "Castle Crags Wilderness",
    "Castle Crags State Park",
    "McArthur–Burney Falls Memorial State Park",
    "LaTour Demonstration State Forest",
    "Mendocino National Forest",
    "Woodson Bridge State Recreation Area",
    "Modoc National Forest",
    "Lava Wilderness",
    "Bucks Lake Wilderness",
    "Plumas National Forest",
    "Sinkyone Wilderness State Park",
    "Westport-Union Landing State Beach",
    "MacKerricher State Park",
    "Jackson Demonstration State Forest",
    "Russian Gulch State Park",
    "Mendocino Woodlands State Park",
    "Navarro River Redwoods State Park",
    "Hendy Woods State Park",
    "Manchester State Park",
    "Salt Point State Park",
    "Fort Ross State Historic Park",
    "Tahoe National Forest",
    "Eldorado National Forest",
    "Mokelumne Wilderness",
    "Humboldt-Toiyabe National Forest"
]
    if len(data_dict["Coordinates"]) < 10:
        #else: print("Looking at next campground..")
        data_list = [str(item).replace('"', '\\"') for item in data_list]
        print("Writing to Google Sheet...")
        worksheet = write_to_google_sheet(SHEET_ID = '1v18uxLmTcApduUgXu7AnxgoMoi6XAy7Lv1Ni9agbxWk', SHEET_NAME = 'Missing Coordinates')
        worksheet.append_rows(data_to_write)
            #time.sleep(1) # Consider adding a delay to ensure you don't hit rate limits
        print("Writing successful! Check Google Sheet.")


    if data_dict["Area_Name"] in covered_area_names:
        #else: print("Looking at next campground..")
        data_list = [str(item).replace('"', '\\"') for item in data_list]
        print("Writing to Google Sheet...")
        worksheet = write_to_google_sheet(SHEET_ID = '1v18uxLmTcApduUgXu7AnxgoMoi6XAy7Lv1Ni9agbxWk', SHEET_NAME = 'Campgrounds in covered areas (potential dups)')
        worksheet.append_rows(data_to_write)
            #time.sleep(1) 
        print("Writing successful! Check Google Sheet.")

    else:
        #else: print("Looking at next campground..")
        data_list = [str(item).replace('"', '\\"') for item in data_list]
        print("Writing to Google Sheet...")
        worksheet = write_to_google_sheet(SHEET_ID = '1v18uxLmTcApduUgXu7AnxgoMoi6XAy7Lv1Ni9agbxWk', SHEET_NAME = 'Main Export Sheet')
        worksheet.append_rows(data_to_write)
            #time.sleep(1) 
        print("Writing successful! Check Google Sheet.")

"""""
#WRITE CAMPSITE NAMES TO GOOGLE SHEET
import time

# Load your namelist from the JSON file
namelist = []
with open("api_pois/CAMPGROUNDS.json") as campground_data_file:
    campgrund_data_dict = json.load(campground_data_file)
    for key, value in islice(campgrund_data_dict.items(), 1001, 2000):
        camp_name_value = value
        facility = camp_name_value[0]
        namelist.append(camp_name_value[0]["FacilityName"])

# Get your worksheet
worksheet = write_to_google_sheet(SHEET_ID='1ekYANpWECKDWGmijUAw59ycT3ni1_P7AmuN9wLy1zmE', SHEET_NAME='All Rec Gov Camp Names')

# Determine the range to update
start_row = 1001
end_row = start_row + len(namelist) - 1

# Prepare the range and values
range_str = f'A{start_row}:A{end_row}'
values = [[name] for name in namelist]

# Update the worksheet in one go
worksheet.update(range_str, values)

print("Writing successful! Check Google Sheet.")

#WRITE FOREST IDS NEXT TO CAMPSITE LIST IN GOOGLE SHEET


forest_id_list = []
with open("api_pois/CAMPGROUNDS.json") as campground_data_file:
    campgrund_data_dict = json.load(campground_data_file)
    for key, value in islice(campgrund_data_dict.items(), 1001, 2000):
        camp_name_value = value
        facility = camp_name_value[0]
        forest_id_list.append(camp_name_value[0]["ParentRecAreaID"])

# Get your worksheet
worksheet = write_to_google_sheet(SHEET_ID='1ekYANpWECKDWGmijUAw59ycT3ni1_P7AmuN9wLy1zmE', SHEET_NAME='All Rec Gov Camp Names')

# Determine the range to update
start_row = 1006
end_row = start_row + len(forest_id_list) - 1

# Prepare the range and values
range_str = f'B{start_row}:B{end_row}'
values = [[forest_id] for forest_id in forest_id_list]

# Update the worksheet in one go
worksheet.update(range_str, values)

print("Writing successful! Check Google Sheet.")
"""