import shutil
import logging
import glob
import os
import yaml
import json
from typing import Dict, Optional
import traceback

from langchain.vectorstores import Chroma

from jarvis.smartgpt import gpt

skill_gen_prompt = """
You are a helpful assistant that writes a json format skill description for the given task and it's execution plan.

You should only respond in JSON format as described below
{
    "skill_name": "get_beijing_weather_of_the_day",
    "skill_description": "Get today's weather in Beijing by browsing the weather websites"
}

Ensure the response can be parsed by Python json.loads"""


def custom_skill_copytree(src_dir, dst_dir):
    os.makedirs(dst_dir, exist_ok=True)

    for file_name in os.listdir(src_dir):
        s = os.path.join(src_dir, file_name)
        d = os.path.join(dst_dir, file_name)

        if os.path.isdir(s):
            continue  # Skip directories

        if file_name.endswith(".yaml") or file_name.endswith(".txt"):
            shutil.copy2(s, d)


class SkillManager:
    name = "skill_saver"
    description = "A skill that saves jarvis skill in a previous step within the skills folder. Not for writing code."

    def __init__(
        self,
        model_name=gpt.GPT_3_5_TURBO_16K,
        retrieval_top_k=5,
        skill_library_dir="skill_library",
    ):
        self.skill_library_dir = skill_library_dir
        self.retrieval_top_k = retrieval_top_k
        self.model_name = model_name
        self.skill_code_dir = f"{skill_library_dir}/code"
        self.skill_vectordb_dir = f"{skill_library_dir}/vectordb"
        self.skill_metadata = f"{skill_library_dir}/skills.json"

        os.makedirs(self.skill_code_dir, exist_ok=True)
        os.makedirs(self.skill_vectordb_dir, exist_ok=True)

        if os.path.exists(self.skill_metadata):
            logging.info(f"Loading Skill Manager from {self.skill_metadata}")
            with open(self.skill_metadata, "r") as file:
                self.skills = json.load(file)
        else:
            self.skills = {}

        embedding_func = gpt.OPEN_AI_MODELS_HUB["text-embedding-ada-002"]
        self.vectordb = Chroma(
            collection_name="skill_vectordb",
            embedding_function=embedding_func,
            persist_directory=self.skill_vectordb_dir,
        )

        assert self.vectordb._collection.count() == len(self.skills), (
            f"Skill Manager's vectordb is not synced with skills.json.\n"
            f"There are {self.vectordb._collection.count()} skills in vectordb but {len(self.skills)} skills in skills.json.\n"
            f"You may need to manually delete the vectordb directory for running from scratch."
        )

    def clone_skill(self, skill_name, dest_dir):
        skill = self.skills.get(skill_name)
        if skill is None:
            raise ValueError(f"Skill '{skill_name}' not found.")

        try:
            custom_skill_copytree(self._skill_dir(skill["skill_name_w_ver"]), dest_dir)
        except Exception as e:
            logging.error("Error saving skill {resp}: {e}")
            logging.info(traceback.format_exc())
            raise e

        return None

    def add_new_skill(self, task_dir, skill_name: Optional[str] = None):
        task, code = self.load_skill_from_dir(task_dir)
        if skill_name is None or len(skill_name) <= 3:
            # use gpt to generate skill name while skill_name is not provided or too short
            skill_name, _ = self.generate_skill_description(task, code)
        skill_description = task

        # prepare skill name and skill dir
        if skill_name in self.skills:
            logging.info(f"Skill {skill_name} already exists. Rewriting!")
            self.vectordb._collection.delete(ids=[skill_name])
            i = 2
            while f"{skill_name}V{i}" in os.listdir(self.skill_code_dir):
                i += 1
            dumped_skill_name = f"{skill_name}V{i}"
        else:
            dumped_skill_name = skill_name

        skill_dir = self._skill_dir(dumped_skill_name)
        logging.info(
            f"generate skill.... skill_dir: {skill_dir}, skill_name: {skill_name}, skill_description: {skill_description}"
        )

        # save skill
        try:
            custom_skill_copytree(task_dir, skill_dir)
        except Exception as e:
            logging.error("Error saving skill {resp}: {e}")
            logging.info(traceback.format_exc())
            raise e

        self.vectordb.add_texts(
            texts=[skill_description],
            ids=[skill_name],
            metadatas=[{"skill_name": skill_name}],
        )
        self.skills[skill_name] = {
            "skill_code": code,
            "skill_description": skill_description,
            "skill_name_w_ver": dumped_skill_name,
        }
        assert self.vectordb._collection.count() == len(
            self.skills
        ), "vectordb is not synced with skills.json"

        with open(self.skill_metadata, "w") as file:
            json.dump(self.skills, file)

        self.vectordb.persist()
        logging.info(f"Saving skill {skill_name} for {task} to {skill_dir}")
        return skill_name

    def generate_skill_description(self, task, code):
        sys_prompt = "Please review the task and its execution plan, and give the task a suitable name\n"
        user_prompt = f"Come up with a detail skill name (skill name should be function-name style, eg. 'get_weather'; skill name should detailed to be unqieu) for the task({task}) execution plan:\n{code}\n###\nSKILL_NAME:"
        skill_name = gpt.complete(
            prompt=user_prompt, model=self.model_name, system_prompt=sys_prompt
        )
        return (skill_name, None)

    def retrieve_skills(self, query):
        k = min(self.vectordb._collection.count(), self.retrieval_top_k)
        if k == 0:
            return {}
        logging.info(f"Skill Manager retrieving for {k} skills")
        try:
            docs_and_scores = self.vectordb.similarity_search_with_score(query, k=k)
        except Exception as e:
            logging.error(f"Error retrieving skills for {query}: {e}")
            return {}

        # Sort docs_and_scores by score in descending order
        logging.info(
            f"Skill Manager retrieved skills: "
            f"{', '.join([doc.metadata['skill_name'] for doc, _ in docs_and_scores])}"
        )

        skills = {}
        for doc, score in docs_and_scores:
            logging.info(
                f"skill: {doc.metadata['skill_name']}, score: {score} for query: {query}"
            )
            skills[doc.metadata["skill_name"]] = {
                "skill_description": self.skills[doc.metadata["skill_name"]][
                    "skill_description"
                ],
                "skill_code": self.skills[doc.metadata["skill_name"]]["skill_code"],
                "match_score": score,
            }

        return skills

    def load_skill_from_dir(self, task_dir):
        plan_file = os.path.join(task_dir, "plan.yaml")
        pwd = os.getcwd()
        logging.info(f"Loading skill from {pwd}/{task_dir}")
        if os.path.exists(plan_file) and os.path.isfile(plan_file):
            execution_plan, code = self.load_yaml(plan_file)
            plan = execution_plan.get("goal")
            if not plan:
                raise ValueError(f"plan not defined in {plan_file}")
            return (plan, code)

        task_files = glob.glob("[0-9]*.yaml", root_dir=task_dir)
        if len(task_files) != 1:
            raise ValueError(
                f"There should be exactly one task file under {task_dir}, but {task_files}"
            )

        task_file = task_files[0]
        execution_plan, code = self.load_yaml(os.path.join(task_dir, task_file))
        task = execution_plan.get("task")
        if not task:
            raise ValueError(f"task is not defined in {task_dir}/{task_file}")
        return (task, code)

    def load_yaml(self, file_name: str) -> (Dict, str):
        try:
            with open(file_name, "r") as stream:
                data = stream.read()
                return yaml.safe_load(data), data
                # return yaml.safe_load(stream), stream.read()
        except Exception as e:
            logging.error(f"Error loading file {file_name}: {e}")
            logging.info(traceback.format_exc())
            raise

    def _skill_dir(self, skill_name):
        return os.path.join(self.skill_code_dir, skill_name)
