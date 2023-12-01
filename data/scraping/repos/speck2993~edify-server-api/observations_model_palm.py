from palm_requests import predict_large_language_model_sample
from gpt_requests import openai_predict_response, get_embedding
import config
import csv, time
import numpy as np

def cosine_similarity(a,b):
    return np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b))

def generate_messages_obs(description):
    starter = '''You are a helpful multilingual workplace safety assistant. Users will provide you with a description of a workplace safety observation and you will extract information from their description. You will sort the observation by category, type, rating, observed party, and location, and communicate this information in English back to the user. The type will be either 'condition' or 'act'. The rating will either be 'safe' or 'unsafe'. The category will be from the following list: Personal Protective Equipment, Theft Preventions, Excavations, Hand/Power Tools, Aerial Lifts/Personnel Baskets, Electrical Cords, Cut/Fill Areas, Traffic Control, Walking/Working Surfaces, Heavy Equipment, Proper Planning And Training, Fall Protection, Trailer/Load Securement, Impalement, Abrasive Blasting, Air Compressors/Hoses, Ladders/Stairways, Team Books, Silica Prevention, Hazard Communication (Container Labeling, Etc), Personnel Baskets, Material Handling, Housekeeping, Specialty Tools, Rigging, Working Over Or Near Water, Fork Lifts/Industrial Powered Trucks, First Aid, Heat Illness Prevention, Scaffolds, Trench/Excavations, Project Information And Postings, Work Permits (Confined Space, Hot Work, Etc), Equipment, Environmental, Stretch & Flex, Cranes And Boom Trucks, Hot Work And Fire Prevention, Lighting/Illumination, Health And Sanitation. The observed party and location will be mentioned explicitly. Anything that doesn't fit these categories can be given as an 'Additional Note'. If not all fields can be filled, fill as much as you can. We've provided some examples to get you started. 
Description: I'm observing Pouring Solutions in Orem, I observed a worker using a ladder to reach a high shelf. The ladder was not secured and the worker was not wearing a hard hat.
Response: Category: Ladders/Stairways, Type: Act, Rating: Unsafe, Observed Party: Pouring Solutions, Location: Orem, Additional Note: none
Description: Gas line struck while digging at the Lebanon site. Operator did not listen to Aaron's directions and dug much deeper than he was suppose to.
Response: Category: Excavations, Type: Act, Rating: Unsafe, Observed Party: none, Location: Lebanon, Additional Note: Please mention the name of the company you are observing.
Description: La escalera está en pleno funcionamiento y no necesita mantenimiento.
Response: Category: Ladders/Stairways, Type: Condition, Rating: Safe, Observed Party: none, Location: none, Additional Note: This message appears to be in Spanish, but was translated to English as 'The ladder is fully functional and does not need maintenance.' Additionally, please mention the company name and site location for your observation.
Description: A worker tried to put out a grease fire with water in the kitchen, burned the goddamn kitchen down, that idiot. Observing Crumbl in Columbus, by the way.    
Response: Category: Hot Work and Fire Prevention, Type: Act, Rating: Unsafe, Observed Party: Crumbl, Location: Columbus, Additional Note: Please refrain from using profanity or directing derogatory insults at workers
Description: I'm in Toronto. Water from runoff and snow melt broke through SWPPP that was in place, observing a Walmart storage facility.
Response: Category: Environmental, Type: Condition, Rating: Unsafe, Observed Party: Walmart, Location: Toronto, Additional Note: none
Description: Observing H&M, just checked on some of the load-securing straps for their shipping trucks.
Response: Category: Trailer/Load Securement, Type: Condition, Rating: none, Observed Party: H&M, Location: none, Additional Note: Not enough information to provide a rating or a location.
Description: Saw a worker doing something unsafe
Response: Category: none, Type: Act, Rating: Unsafe, Observed Party: none, Location: none, Additional Note: Please provide more information.
Description: Kana Pipeline in Riverside has trenches sloped at the proper angle
Response: Category: Trench/Excavations, Type: Condition, Rating: Safe, Observed Party: Kana, Location: Riverside, Additional Note: none
Description: '''
    starter = starter + description + '\nResponse: '
    return starter

def observation_response(description):
    prompt = generate_messages_obs(description)
    response = predict_large_language_model_sample(config.gci_project_name, "text-bison@001", 0.2, 256, 0.8, 40, prompt, "us-central1")
    return response

def generate_messages_core_obs(description):
    starter = '''You are a helpful multilingual workplace safety assistant. Users will provide you with a description of a workplace safety observation, and you will extract information from their description to fill out a form. You will sort the observation by category, type, division, and project. The possible types are: Near Miss, Unsafe Act, Recommendation, and Recognition. The category will be one of the following: Gases/Flammables/Combustibles, Competent Person, Confind Space, Cranes/Rigging/Inspections, IDLH (Immediately Dangerous to Life or Health), Electrical Safety - Low Voltage, Electrical Safety - High Voltage, Stormwater Pollution Prevention, Excavations, Fall Procection, Fire Protection, First Aid, Floor Holes, Forklifts/Elevating Platforms/Aerial Devices, Guardrails/Toe boards, HazCom, Heavy Construction Equipment, Heat Illness Prevention, Hot Work/Welding, Housekeeping, Ladders, Material Handling, PPE, Posting Requirements, Respiratory Protection/Use/Storage, Scaffolding, Struck By/Caught In/Caught Between, Tools and Equipment, Traffic Control, Training, Health Hazards, Impalement Hazard, Machine Guarding, Access and Egress, Lighting, Lockout/Tagout, Toilets/Sanitation, Tunnels and Tunneling. The project and division are the name of the site and the name of the team working on that specific task. If not all fields can be filled, fill as much as you can. You will also provide an 'Additional Note' field for any information that doesn't fit these fields, or for additional possiblities for the category of the observation. If any field is not specifically indicated in the prompt, use 'none' as the response for that field. Here are some examples to get you started.
Description: I'm observing Pouring Solutions in Orem on Project 1032, I watched a worker using a ladder to reach a high shelf. The ladder was not secured and the worker was not wearing a hard hat.
Response: Category: Ladders, Type: Unsafe Act, Project: Project 1032, Division: Pouring, Additional Note: Category may also be 'PPE' or 'Fall Protection'
Description: Fire escapes are blocked by storage and are not accessible at Littlerock.
Response: Category: Access and Egress, Type: Unsafe Act, Project: Littlerock, Division: none, Additional Note: none
Description: Sufficient ventilation is being used to protect from particulates when angle grinding at the Tire place.
Response: Category: Respiratory Protection/Use/Storage, Type: Recognition, Project: Tire, Division: HVAC, Additional Note: none
Description: Walkie-talkies are in good condition and are used when coordinating crane operations on the Gordon
Response: Category: Cranes/Rigging/Inspections, Type: Recognition, Project: Gordon, Division: Crane Operations/Communication Equipment, Additional Note: none
Description: Radioactive materials aren't stored properly, these fucking idiots could have killed someone.
Response: Category: Immediately Dangerous to Life or Health, Type: Unsafe Act, Project: none, Division: none, Additional Note: Please speak professionally for more accurate interpretation. Category may also be 'Material Handling'.
Description:  
'''
    starter = starter + description + "\nResponse: "
    return starter

def core_obs_response(description):
    prompt = generate_messages_core_obs(description)
    response = predict_large_language_model_sample(config.gci_project_name,"text-bison@001", 0.2, 256, 0.8, 40, prompt, "us-central1")
    return response

def get_project(project,projects):
    starter = '''We're trying to identify which construction project a certain team is working on. We have an informal name for the project and a list of possible official names. Pick the official name that best fits the informal name. Your answer should be one line and should exactly match one of the possible names. If it doesn't match any name, reply 'none'.
Informal Name: ''' + str(project) + '''
Possible Names: '''
    for p in projects:
        starter = starter + p + "\n"
    starter = starter + "Official name: "
    response = openai_predict_response(starter)
    return response

def get_division(obs,division,divisions):
    starter = '''We're trying to identify which construction division a supervisor is observing based on what they said during their observation. We have an informal guess for the division and a list of possible official names. Pick the official name that best fits the informal name. Your answer should be one line and should exactly match one of the possible names. If it doesn't match any name, reply 'none'.
Observation: ''' + str(obs) + '''
Informal Name: ''' + str(division) + '''
Possible Names:\n'''
    for d in divisions:
        starter = starter + d + "\n"
    starter = starter + "Official name: "
    response = openai_predict_response(starter)
    return response

def get_contractors(contractor,contractor_names,contractor_embeddings):
    #Compute an embedding for a sentence involving the contractor
    #Compute the cosine similarity between the embedding and the embeddings of the contractors
    #Return the contractors with the 10 highest cosine similarities
    #Ask the user to pick the contractor from the list
    embedded_sentence = "The contractor name is [CN]."
    embedded_sentence = embedded_sentence.replace("[CN]",contractor)
    emb = get_embedding(embedded_sentence)
    #read in contractors file. First column is contractor name, second column is embedding
    #skip the first row
    contractors = []
    embeddings = np.load(contractor_embeddings)
    with open(contractor_names) as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            contractors.append(row[0])
    #compute cosine similarity between each contractor embedding and the sentence embedding
    scores = []
    for i in range(len(embeddings)):
        scores.append(cosine_similarity(emb,embeddings[i]))
    #sort contractors by cosine similarity
    scores = np.array(scores)
    contractors = np.array(contractors)
    sorted_contractors = contractors[np.argsort(scores)]
    sorted_scores = scores[np.argsort(scores)]
    #add the contractors in the top 6 to the return list
    return_list = []
    for i in range(10):
        return_list.append(sorted_contractors[-i])
    
    #now add any contractors for which the contractor abbreviation could be an acronym
    #for example, if the contractor is "ABC Construction", the abbreviation could be "ABC" or "AC"
    #if the abbreviation is "AC", we want to add "ABC Construction" to the list of possible contractors
    #we do this with a simple heuristic: if the abbreviation is a substring of the contractor name, add the contractor name to the list
    for i in range(len(contractors)):
        if contractor in contractors[i]:
            return_list.append(contractors[i])

    #remove duplicates
    return_list = list(set(return_list))

    return return_list


def generate_message_facebook_obs(description):
    starter = '''You are a helpful multilingual workplace safety assistant. Users will provide you with a description of a workplace safety observation, and you will extract information from their description to fill out a form. You will sort the observation by division, project, location, category, type, rating, and you will also indicate whether the issue was addressed on the spot. Location will further be broken down into building, area, level, and room. The possible categories are: Abrasive Blasting, Air and Powder Actuated Tools, Air Compressors, Barricades/Access Control, Body/Working Position (Ergonomics), Caught In-Between, Chemical Hazard, Commissioning Activity, Concrete Pumping, Confined Space, Covid Communication, Covid PPE, Covid Safe Distancing, Covid Safe Entry and Screening, Covid Sanitation, Covid Transportation, Covid Work Arrangement, Covid Workforce Zoning, Cranes and Boom Trucks, Creature Hazard, Cut/Laceration Hazard, Documentation, Electrical Hazard, Electrical Tools and Cords, Emergency Access or Egress, Emergency Response or Employee Care Related, Environmental Hazard, Excavations, Fall Protection, Falling Object Protection, Hot Work/Fire Protection, Forklifts, Fuel and Chemical Storage, GHS/HAZCOM, Hand Safety/positioning, Hand Tools, Heat Stress, Heavy Equipment, Housekeeping, Human Factor-Behaviours, Impalement Hazard, Industrial Hygiene (Noise/Air), Ladders and Stairways, Lighting, Lockout/Tag Out, Machine Guarding, Material Handling, Material Storage, Mobile Elevating Work Platform, Mobile Plant, New Hire Program, Overhead Work, Personal Protective Equipment, Pinch Point Hazard, Puncture Hazard, Regulatory Postings and Signage, Rigging Materials and Methods, Sanitation (Health), Scaffolds and Temporary Work Platforms, Security, Signage, Struck By, Temporary Power, Temperature, Trade Damage, Traffic or Pedestrian Controls, Walking and Working Surfaces, Welding and Cutting, Wellness and Wellbeing. The possible types are: Act and Condition. The possible ratings are: Opportunity for improvement and Positive reinforcement. Unless specifically indicated in the description, reply 'N/A' for the 'Corrected' field.'''
    starter = starter + ''' The possible buildings are: Building 1, Building 2, Building 3, Building 4, Building 5, Building 6, Building 7, Administration Building, Batch Plants, Construction Locations, Entrance, Ibos, Main Building, Offsite Facility, PEMB/Construction Offices, Site Office, TOL 1/Plot 1 Carpark, TOL 2, Worker Experience Yard. The possible areas are: A, B, C, D, Admin (F), Core (E), Conveyance Pipe - Cole Road, Conveyance Pipe - Curtis Road, Conveyance Pipe - Kuna Mora Road, Employee Parking, Gantry, Generator Yard, Influent Lift Station, Infrastructure, KE01, KE02, KE03, Lagoons, Laydown, Lunch Area, NCB, North 40, Offices, Operations Yard, Phase 1B, Phase 1C, Restroom, Roof, SGA 1 (Level 9/10), Site, Site substation. The possible levels are: Underground, Ground, Penthouse, Roof, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, Northwest Gantry, Northeast Gantry, Southwest Gantry, Southeast Gantry.'''
    starter = starter + ''' The possible rooms are: Coil Damper Room, Coil Room, Core Wall 1, Core Wall 2, Core Wall 3, Core Wall 4, Core Wall 5, Core Wall 6, Core Wall 7, Core Wall 8, Core Wall 9, Core Wall 10, Core Wall 11, Corridor, Data Hall, Detention Tank 1, Detention Tank 2, Detention Tank 3, Dump Tank 1, Dump Tank 2, Dump Tank 3, E. Coil Vestibule, E. Coil Damper Vestibule, E. Evaporator Vestibule, E. Fan Vestibule, E. Filter Vestibule, Electrical Room, Evaporator Room, Exhaust Room, Fan Room, Filter Room, Fuel Tank, Intake Room, Loading Dock, MDF, Network Core, North Data Hall, North Electrical Room, Process Water Plant (Pwp), Relief Fan Room, Remote - IDF, South Electrical Room, South Data Hall, Stair 1, Stair 2, Storage, Temporary Substation, Vestibule, W. Coil Damper Vestibule, W. Coil Vestibule, W. Evaporator Vestibule, W. Fan Vestibule, W. Filter Vestibule, Water Treatment, Water Tanks, 22kv Substation, 66kv Substation.'''
    starter = starter + ''' It's extremely important that you only provide information from these lists. If not all fields can be filled, fill as much as you can. Here are some examples to get you started.'''
    starter = starter + '''\nDescription: Over at Building 3, Area B on the 3rd floor, there was this situation. Workers from TechVent HVAC had tools spread everywhere on the Greenview build, real tripping hazards. Pointed it out and they tidied up fast.'''
    starter = starter + '''\nResponse: Division: HVAC, Project: Greenview, Building: Building 3, Area: B, Level: none, Room: 3, Category: Material Storage, Contractor: TechVent HVAC, Type: condition, Rating: Opportunity for improvement, Corrected: Y'''
    starter = starter + '''\nDescription: In Building 4, I spotted the guys from ElectraWorx dealing with some electrical tools. No cords were labeled, so I wasn't sure which were live. Needs to be sorted out soon.'''
    starter = starter + '''\nResponse: Division: Electrical, Project: none, Building: Building 4, Area: none, Level: none, Room: none, Category: Electrical Tools and Cords, Contractor: ElectraWorx, Type: condition, Rating: Opportunity for improvement, Corrected: N'''
    starter = starter + '''\nDescription: At the Laydown Area in Building 5 for Oceanview Condows, some materials from BuildFast were spread everywhere. Looked chaotic.'''
    starter = starter + '''\nResponse: Division: Civil, Project: Oceanview Condos, Building: Building 5, Area: Laydown, Level: none, Room: none, Category: Material Storage, Contractor: BuildFast, Type: condition, Rating: Opportunity for improvement, Corrected: N/A'''
    starter = starter + '''\nDescription: Hey, noticed missing guardrails on the third level scaffolding and some worn-out boards on the work platform for the framing team at the high school project with Union General. No toe boards either.'''
    starter = starter + '''\nResponse: Division: Framing, Project: High School, Building: none, Area: Work Platform, Level: Third Level, Room: none, Category: Scaffolds and Temporary Work Platforms, Contractor: Union General, Type: condition, Rating: Opportunity for improvement, Corrected: N/A'''
    starter = starter + '''\nDescription: ''' + description + '''\nResponse: '''

    return starter

def facebook_obs_response(description):
    prompt = generate_message_facebook_obs(description)
    response = predict_large_language_model_sample(config.gci_project_name,"text-bison@001", 0.2, 256, 0.8, 40, prompt, "us-central1")
    return response