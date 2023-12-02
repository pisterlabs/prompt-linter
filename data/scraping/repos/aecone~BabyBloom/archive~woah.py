import cohere
import numpy as np
import os
from dotenv import load_dotenv
from archive.profiles import moms, nurses

def get_mom():
    while True:
        mom_id = input("What's your ID number? ('exit' to quit) ")

        if mom_id.lower() == 'exit':
            print("Exiting the program.")
            return None

        for profile_id, profile in moms.items():
            if profile.get('id') == mom_id:
                nursing_tasks = ' '.join(map(str, profile.get('nursing_tasks')))
                # print(f"Nursing tasks for mom with ID {mom_id}: {nursing_tasks}")
                return nursing_tasks
            else:
                print("The ID does not exist. Please re-enter your ID.")

#Get nursing tasks from a profile and join them into a single string.
def get_nurse():
    all_nurses_tasks = []
    for nurse_profile in nurses.values():
        nurse_profile_task = ' '.join(nurse_profile.get('nursing_tasks', []))
        all_nurses_tasks.append(nurse_profile_task)
    # print(all_nurses_tasks)
    return all_nurses_tasks

#Combinne nursing tasks from moms and nurses. Make embeddings for each profile 
def get_embeddings(mom_tasks, nurse_tasks):
    similarity_dict = {}
    mom_nurse_tasks = [mom_tasks] + nurse_tasks
    # print(mom_nurse_tasks)
    co = cohere_api()
    # Fetch embeddings for the nursing tasks using cohere API
    num_profiles = tuple(i for i in range(len(mom_nurse_tasks)))
    num_profiles = co.embed(
        model='embed-english-v2.0',
        texts=mom_nurse_tasks ).embeddings
      # Calculate similarity for each pair of embeddings
    for i in range( len(num_profiles)-1):
        # Get the nurse ID associated with this index
        nurse_key = list(nurses.keys())[i]  # Get the nurse ID (e.g., 'nurse1', 'nurse2')
        nurse_id = nurses[nurse_key]['id'] 
        # Calculate similarity and store in the dictionary with nurse ID as key
        similarity_dict[nurse_id] = calculate_similarity(num_profiles[0], num_profiles[i])
    return similarity_dict

#Get API key and Initialize the Cohere client
def cohere_api():
    load_dotenv()
    API_KEY = os.getenv('COHERE_API_KEY')
    return cohere.Client(API_KEY)

#Calculate similarity between two embeddings
def get_best_embed(similarity_embed):
    max_similarity_id = max(similarity_embed, key=lambda k: similarity_embed[k])
    max_nurse_name = get_nurse_name(max_similarity_id)
    print(f"The nurse with the maximum similarity is {max_nurse_name} with the " + str(similarity_embed[max_similarity_id]))

def get_nurse_name(nurse_id):
    for nurse_key, nurse_profile in nurses.items():
        if nurse_profile['id'] == nurse_id:
            return nurse_profile['name']
    return None

def calculate_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def main():
    mom_tasks = get_mom()
    nurse_tasks = get_nurse()
    similarity_embed = get_embeddings(mom_tasks, nurse_tasks)
    get_best_embed(similarity_embed)

if __name__ == "__main__":
    main()
