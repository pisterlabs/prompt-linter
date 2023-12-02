import numpy as np
import openai
from pymongo import MongoClient
from Config import Config
from Backend.Action import Action
from sklearn.metrics.pairwise import cosine_similarity

config = Config()

openai.api_key = config["OPENAI_API_KEY"]


class MongoManager:
    def __init__(
        self,
        mongo_url=config["MONGO_DB_URL"],
        db_name=config["DB_NAME"],
    ):
        self.client = MongoClient(mongo_url)
        self.db = self.client[db_name]
        self.collection = self.db["Users"]
        self.approved_skills = self.db["Approved_Skills"]
        self.collection.create_index("uid", unique=True)
    
    def get_approved_skills(self, uid):
        return self.approved_skills.find({"approved": True})     
            
    def insert_document(self, document):
        return self.collection.insert_one(document)

    def get_valid_uids(self):
        return [doc["uid"] for doc in self.collection.find()]

    def get_user_installed_skills(self, uid):
        user_doc = self.collection.find_one({"uid": uid})
        return user_doc.get("installed_skills", [])

    def get_user_installed_actions(self, uid):
        user_doc = self.collection.find_one({"uid": uid})
        return [
            {key: value for key, value in action.items() if key != "vector"}
            for action in user_doc.get("actions")
        ]

    def generate_embedding(self, text):
        try:
            response = openai.Embedding.create(model="text-embedding-ada-002", input=text)
        except:
            openai.api_key = config["OPENAI_API_KEY"]
            return self.generate_embedding(text)
        return np.array(response["data"][0]["embedding"])

    def add_actions_to_user(self, user_id, actions: list[Action]):
        for action in actions:
            self.add_action_to_user(user_id, action)

    def add_action_to_user(self, user_id, action: Action):
        user_doc = self.collection.find_one({"uid": user_id})

        # create name vector
        name_vector = self.generate_embedding(action.name)

        # create skill vector
        skill_vector = np.zeros_like(name_vector)
        skill_token = self.generate_embedding(action.skill)
        skill_vector += skill_token

        # combine the vectors
        combined_vector = name_vector + skill_vector

        for parameter in action.parameters:
            string_format = f"{parameter.id}: {parameter.description}"
            parameter_vector = np.zeros_like(combined_vector)
            parameter_token = self.generate_embedding(string_format)

            parameter_vector += parameter_token

            combined_vector = combined_vector + parameter_vector

        combined_vector = combined_vector.tolist()

        if user_doc:
            if "actions" in user_doc and any(
                s["id"] == action.id for s in user_doc["actions"]
            ):
                self.collection.update_one(
                    {"uid": user_id, "actions.id": action.id},
                    {
                        "$set": {
                            "actions.$": {"vector": combined_vector, **action.to_dict()}
                        }
                    },
                )
            else:
                self.collection.update_one(
                    {"uid": user_id},
                    {
                        "$push": {
                            "actions": {"vector": combined_vector, **action.to_dict()}
                        }
                    },
                )
        return None
    
    def remove_skill_from_user(self, skill_name, user_id):
        user_doc = self.collection.find_one({"uid": user_id})

        if user_doc:
            # Remove the skill from the installed_skills list
            updated_installed_skills = [
                skill for skill in user_doc.get("installed_skills", []) if skill["name"] != skill_name
            ]

            # Remove all actions with the specified skill name
            updated_actions = [
                action for action in user_doc.get("actions", []) if action.get("skill") != skill_name
            ]

            self.collection.update_one(
                {"uid": user_id},
                {
                    "$set": {
                        "installed_skills": updated_installed_skills,
                        "actions": updated_actions,
                    }
                },
            )
        return None
    
    def add_skill_to_user(self, name, version, user_id, url):
        user_doc = self.collection.find_one({"uid": user_id})

        if user_doc:
            if "installed_skills" in user_doc and any(
                s["name"] == name for s in user_doc["installed_skills"]
            ):
                self.collection.update_one(
                    {"uid": user_id, "installed_skills.name": name},
                    {
                        "$set": {
                            "installed_skills.$": {"name": name, "version": version, "url": url}
                        }
                    },
                )
            else:
                self.collection.update_one(
                    {"uid": user_id},
                    {"$push": {"installed_skills": {"name": name, "version": version, "url": url}}},
                )
        return None

    def get_user_actions(self, user_id):
        user_doc = self.collection.find_one({"uid": user_id})
        return (
            [Action.from_dict(skill) for skill in user_doc.get("actions", [])]
            if user_doc
            else []
        )

    def get_relavent_actions(self, search_term, user_id, limit=None):
        user_doc = self.collection.find_one({"uid": user_id})

        if not user_doc:
            return []

        actions = {
            action["id"]: {
                "vector": action["vector"],
                "action": Action.from_dict(action),
            }
            for action in user_doc.get("actions", [])
        }

        search_vector = self.generate_embedding(search_term).reshape(1, -1)

        similarity_scores = {}
        for id, entry in actions.items():
            similarity = cosine_similarity(
                search_vector, np.array(entry["vector"]).reshape(1, -1)
            )
            similarity_scores[id] = similarity[0][0]

        sorted_entries = sorted(
            similarity_scores.items(), key=lambda x: x[1], reverse=True
        )
        closest_entries = sorted_entries[:limit]

        return [actions[entry[0]]["action"] for entry in closest_entries]

    def clear_collection(self):
        self.collection.delete_many({})
    
    def add_user(self, uid):
        """
        Add a new user to the database.

        Args:
            uid (str): A uid to add

        Returns:
            None
        """
        # Check if the user already exists in the database by their unique ID
        existing_user = self.collection.find_one({"uid": uid})

        if existing_user is None:
            self.insert_document({"uid": uid})
