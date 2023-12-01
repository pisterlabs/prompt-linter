import cohere
import numpy as np
import os
from dotenv import load_dotenv


class NursingProfileManager:
    def __init__(self, moms, nurses):
        self.moms = moms
        self.nurses = nurses

    def get_mom(self):
        while True:
            mom_id_input = input("\nWhat's your ID number? ('exit' to quit) ")

            if mom_id_input.lower() == 'exit':
                print("\nExiting the program.")
                return None

            # Iterate through mom profiles
            for mom_key, mom_profile in self.moms.items():
                if mom_profile.id == mom_id_input:
                    nursing_tasks = ' '.join(
                        map(str, mom_profile.nursing_tasks))
                    mom_nurse_tasks = self.get_mom_tasks(mom_profile.id)
                    mom_nursing_tasks_string = ", ".join(mom_nurse_tasks)
                    print(
                        f"\nYou have claimed that you want your nurse to do {mom_nursing_tasks_string}.")
                    return nursing_tasks
            else:
                print("\nThe ID does not exist. Please re-enter your ID.")

    def get_nurse(self):
        all_nurses_tasks = []
        # Iterate through nurse profiles
        for nurse_key, nurse_profile in self.nurses.items():
            nurse_profile_task = ' '.join(nurse_profile.nursing_tasks)
            all_nurses_tasks.append(nurse_profile_task)
        return all_nurses_tasks

    def get_embeddings(self, mom_tasks, nurse_tasks):
        # Combine mom and nurse tasks
        mom_nurse_tasks = [mom_tasks] + nurse_tasks
        # print(mom_nurse_tasks)
        co = self.cohere_api()
        num_profiles = tuple(i for i in range(len(mom_nurse_tasks)))
        # print(num_profiles)
        num_profiles = co.embed(
            model='embed-english-v2.0',
            texts=mom_nurse_tasks).embeddings
        similarity_dict = {}
        # Calculate similarity for each nurse
        for i in range(len(nurse_tasks)):
            nurse_key = list(self.nurses.keys())[i]
            nurse_id = self.nurses[nurse_key].id
            similarity_dict[nurse_id] = self.calculate_similarity(
                num_profiles[0], num_profiles[i+1])

        return similarity_dict

    def cohere_api(self):
        # Load API key from environment and initialize Cohere client
        load_dotenv()
        API_KEY = os.getenv('COHERE_API_KEY')
        return cohere.Client(API_KEY)

    def calculate_similarity(self, a, b):
        # Calculate cosine similarity between embeddings
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    def get_best_embed(self, similarity_embed):
        # Find nurse with maximum similarity
        max_similarity_id = max(
            similarity_embed, key=lambda k: similarity_embed[k])
        max_nurse_name = self.get_nurse_name(max_similarity_id)
        max_nurse_tasks = self.get_nurse_tasks(max_similarity_id)
        nursing_tasks_string = ", ".join(max_nurse_tasks)
        print(
            f"\nThe nurse that is best fit for you is {max_nurse_name} with a similarity score of {similarity_embed[max_similarity_id] * 100:.2f}%.")
        print(f"\nThe nurse is qualified for {nursing_tasks_string}.\n")
        return max_similarity_id

#TODO: Make for oop
    def get_nurse_name(self, nurse_id):
        # Get nurse name based on ID
        for nurse_key, nurse_profile in self.nurses.items():
            if nurse_profile.id == nurse_id:
                return nurse_profile.name
        return None

    def get_nurse_tasks(self, nurse_id):
        # Get nurse name based on ID
        for nurse_key, nurse_profile in self.nurses.items():
            if nurse_profile.id == nurse_id:
                return nurse_profile.nursing_tasks
        return None

    def get_mom_tasks(self, nurse_id):
        # Get nurse name based on ID
        for mom_key, mom_profile in self.moms.items():
            if mom_profile.id == nurse_id:
                return mom_profile.nursing_tasks
        return None
