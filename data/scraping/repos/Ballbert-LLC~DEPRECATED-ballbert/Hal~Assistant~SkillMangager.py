import ast
import json
import pickle
import socket
import openai
import inspect
import shutil
import sqlite3
import uuid


import yaml
from git import Repo
import importlib

from ..Logging import log_line

from . import Assistant
from ..Decorators import reg

import os
from Config import Config
from ..Utils import rmtree_hard

config = Config()

repos_path = f"{os.path.abspath(os.getcwd())}/Skills"

NoneType = type(None)


def deepcopy(original_dict):
    """
    Performs a deep copy of a dictionary.

    Args:
        original_dict (dict): The original dictionary to be copied.

    Returns:
        dict: A deep copy of the original dictionary.
    """
    copied_dict = {}
    for key, value in original_dict.items():
        if isinstance(value, dict):
            copied_dict[key] = deepcopy(value)
        elif isinstance(value, list):
            copied_dict[key] = value.copy()
        else:
            copied_dict[key] = value
    return copied_dict


class SkillMangager:
    def __init__(self) -> None:
        pass

    def get_actions_dict(self):
        action_dict: dict = reg.all
        for skill_id, item in action_dict.items():
            _parameters = action_dict[skill_id]["docstring"].params

            action_dict[skill_id]["parameters"] = (
                action_dict[skill_id]["parameters"]
                if "parameters" in action_dict[skill_id]
                else {}
            )

            for param in _parameters:
                _name = param.arg_name
                _type = param.type_name

                _description = param.description
                _required = not param.is_optional

                action_dict[skill_id]["parameters"][_name] = {
                    "type": _type,
                    "description": _description,
                    "required": _required,
                }

            if "self" in action_dict[skill_id]["parameters"]:
                del action_dict[skill_id]["parameters"]["self"]
        return action_dict

    def get_class_function(self, class_instance):
        class_functions = dict()

        for name, value in inspect.getmembers(class_instance):
            is_method = inspect.ismethod(value) and hasattr(value, "__func__")
            # Check if the function has the reg Decorator
            # First goes through every element of the class and it checks if it is a function. It then checks if that function is also in the dict of every function with the decorator
            if is_method and name in [
                value["func_name_in_class"] for key, value in reg.all.items()
            ]:
                class_functions[name] = value.__func__

        return class_functions

    def update_actions_function(self, class_functions, action_dict, class_instance):
        new_action_dict = deepcopy(action_dict)

        for func_name, func in class_functions.items():
            for action_name, values in new_action_dict.items():
                new_values = deepcopy(values)
                if func_name == new_values["func_name_in_class"]:
                    # factory is in order to loose referance to the sub function

                    def wrapper_func_factory(func):
                        def wrapper_func(*args, **kwargs):
                            signature = inspect.signature(func)
                            if len(signature.parameters) > 0:
                                return func(class_instance, *args, **kwargs)
                            else:
                                return func(*args, **kwargs)

                        return wrapper_func

                    new_values["function"] = wrapper_func_factory(func)
                    new_values["class_instance"] = class_instance
                    new_action_dict[action_name] = new_values

        return new_action_dict

    def add_skill(self, assistant, skill):
        """Adds a skill to the memeory

        Keyword arguments:
        assistant -- Assistant instance
        assistant -- Name of skill
        Return: return_description
        """

        prev_actions_dict = deepcopy(assistant.action_dict)

        module = importlib.import_module(f"Skills.{skill}")

        desired_class = getattr(module, skill)

        action_dict = self.get_actions_dict()

        action_dict = self.get_new_actions_from_current_actions(
            action_dict, prev_actions_dict
        )

        image = self.get_image_url(skill=skill, action_dict=action_dict)

        class_instance = desired_class()

        class_functions = self.get_class_function(class_instance)

        assistant.installed_skills[skill] = {
            "image": image,
            "name": skill,
            "version": 0.0,
            "actions": class_functions,
            "class": class_instance,
        }

        action_dict = self.update_actions_function(
            class_functions, action_dict, class_instance
        )

        for key, value in action_dict.items():
            assistant.action_dict[key] = value

    def get_new_actions_from_current_actions(self, new_actions_dict, old_actions_dict):
        updated_actions = {}
        for key, _ in new_actions_dict.items():
            if key not in old_actions_dict:
                updated_actions[key] = new_actions_dict[key]

        return updated_actions

    def get_image_url(self, skill, action_dict):
        if os.path.exists(os.path.join(repos_path, skill, "icon.png")):
            shutil.copyfile(
                os.path.join(repos_path, skill, "icon.png"),
                os.path.join(
                    os.path.abspath("./"),
                    "Flask",
                    "static",
                    "images",
                    f"{skill}_icon.png",
                ),
            )

            IPAddr = socket.gethostbyname(socket.gethostname())

            image = f"http://{IPAddr}:5000/images/{skill}_icon.png"
        else:
            res = ""

            description = res or None

            with open(f"{repos_path}/{skill}/config.yaml", "r") as stream:
                data_loaded = yaml.safe_load(stream)
                name = data_loaded["name"]
                description = (
                    data_loaded["description"]
                    if "description" in data_loaded
                    else "No description provided"
                )

            des = ""
            for id, value in action_dict.items():
                if name == value["skill"]:
                    des += f"      Name: {value['name']}, Paramiters: {value['parameters']}\n"

            result = f"""

                Name of skill:
                    {name}

                Description of skill:
                    {description}

                Actions:
            {des}

            """

            prompt = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a prompt generator for an image generator that. Your prompts are made to be an turn into an icon for a skill addon with the details in the paramiters the user provides. Constraints: Speak in language that an image generator could understand.",
                    },
                    {"role": "user", "content": result},
                ],
            )

            prompt = prompt["choices"][0]["message"]["content"]

            openai.api_key = config["OPENAI_API_KEY"]

            response = openai.Image.create(prompt=prompt, n=1, size="1024x1024")

            image_url = response["data"][0]["url"]
            image = image_url
        return image

    def get_name(self):
        with open(f"{repos_path}/temp/config.yaml", "r") as stream:
            data_loaded = yaml.safe_load(stream)

            name = data_loaded["name"] if data_loaded["name"] else uuid.uuid4()
        return name

    def get_requirements(self):
        with open(f"{repos_path}/temp/config.yaml", "r") as stream:
            data_loaded = yaml.safe_load(stream)
            requirements = (
                data_loaded["requirements"] if "requirements" in data_loaded else []
            )
        return requirements

    def ready_temp_dir(self):
        # make sure temp dir exists and is empty
        if not os.path.exists(os.path.join(repos_path, "temp")):
            os.makedirs(f"{repos_path}/temp")
        else:
            rmtree_hard(os.path.join(repos_path, "temp"))
            os.makedirs(f"{repos_path}/temp")

    def rename_temp(self):
        temp_name = uuid.uuid4()
        os.rename(
            os.path.join(repos_path, "temp"), os.path.join(repos_path, str(temp_name))
        )
        return temp_name

    def install_requirements(self, assistant, requirements, name):
        requirements_names = []
        updated_requirements = requirements.copy()
        for requirement in requirements:
            req_name = self.add_skill_from_url(assistant, requirement)
            if req_name != False:
                requirements_names.append(req_name)
            else:
                updated_requirements.remove(requirement)
        return requirements_names

    def check_if_in_sql(self, name):
        con = sqlite3.connect("skills.db")
        cur = con.cursor()

        cur.execute(
            "SELECT EXISTS(SELECT 1 FROM installedSkills WHERE skill = ?)", (name,)
        )

        # Fetch the result
        result = cur.fetchone()[0]

        con.commit()
        con.close()

        return True if result else False

    def rename_to_perm_name(self, name, temp_name):
        # check if it is already installed
        if os.path.exists(f"{repos_path}/{name}"):
            rmtree_hard(f"{repos_path}/{temp_name}")
            print("Already Installed " + name)
            raise Exception("Already Installed")

        # Save it is name
        os.rename(f"{repos_path}/{temp_name}", f"{repos_path}/{name}")

        self.check_if_in_sql(name)

        # Make sure it is valid
        if not self.is_folder_valid(f"{repos_path}/{name}"):
            rmtree_hard(os.path.join(repos_path, name))
            print("invalid")
            raise Exception("Invallid Package")

    def create_settings_meta(self, name):
        open(f"{repos_path}/{name}/settings.yaml", "w").close()

    def dump_meta_to_yaml(self, name):
        if os.path.exists(f"{repos_path}/{name}/settingsmeta.yaml"):
            with open(f"{repos_path}/{name}/settingsmeta.yaml", "r") as file:
                data = yaml.safe_load(file)
            settings = {}
            if data and "skillMetadata" in data and "sections" in data["skillMetadata"]:
                for section in data["skillMetadata"]["sections"]:
                    for field in section["fields"]:
                        if "name" in field and "value" in field:
                            field_name = field["name"]
                            value = field["value"]
                            if value == "true" or value == "True":
                                value = True
                            elif value == "false" or value == "False":
                                value = False
                            if not isinstance(value, NoneType):
                                settings[field_name] = value
                            else:
                                settings[field_name] = None
                with open(f"{repos_path}/{name}/settings.yaml", "w") as file:
                    yaml.dump(settings, file)
            else:
                open(f"{repos_path}/{name}/settings.yaml", "w").close()

    def get_new_actions(self, assistant, prev_action_dict):
        new_action_dict = {}

        for skill_id, item in assistant.action_dict.items():
            if skill_id not in prev_action_dict:
                new_action_dict[skill_id] = item

        return new_action_dict

    def clear_db(self, name):
        con = sqlite3.connect("skills.db")
        cur = con.cursor()
        cur.execute(f"DELETE FROM actions WHERE skill='{name}'")
        con.commit()
        con.close()

    def insert_actions(self, result, name):
        con = sqlite3.connect("skills.db")
        cur = con.cursor()

        for action_id, action in result.items():
            cur.execute(
                f"""
            INSERT INTO actions VALUES
                ('{name}', '{action["uuid"]}' ,'{action["id"]}', '{action["name"]}',
                    '{str(action["parameters"]).replace("'", '"')}')
            """
            )
        con.commit()
        con.close()

    def add_to_installed_skills(self, name):
        con = sqlite3.connect("skills.db")
        cur = con.cursor()

        cur.execute(
            f"""
                INSERT INTO installedSkills (skill, version)
                VALUES ('{name}', 0.0)
        """
        )

        con.commit()
        con.close()

    def add_skill_from_url(self, assistant: Assistant, url: str):
        try:
            self.ready_temp_dir()
            Repo.clone_from(url, f"{repos_path}/temp", depth=1)
        except Exception as e:
            log_line("Err", e)
            raise e
        try:
            name = self.get_name()
            requirements = self.get_requirements()
            temp_name = self.rename_temp()
            requirements_names = self.install_requirements(
                assistant, requirements, name
            )
            prev_action_dict: dict = deepcopy(assistant.action_dict)
        except Exception as e:
            rmtree_hard(f"{repos_path}/temp")
            log_line("err", e)
            raise e
        try:
            self.rename_to_perm_name(name, temp_name)
        except Exception as e:
            log_line("err", e)
            raise e
        try:
            self.create_settings_meta(name)
            self.dump_meta_to_yaml(name)
            self.add_skill(assistant, name)
            new_action_dict: dict = self.get_new_actions(assistant, prev_action_dict)
        except Exception as e:
            rmtree_hard(f"{repos_path}/{name}")
            log_line("err", e)
            raise e
        try:
            actions_with_uuids = assistant.pm.add_list(new_action_dict, name)
        except Exception as e:
            actions_with_uuids = []
            self.remove_from_memory(name, assistant)
            rmtree_hard(f"{repos_path}/{name}")
        try:
            self.insert_actions(actions_with_uuids, name)
            self.add_to_installed_skills(name)
            self.add_requirements_to_db(requirements, requirements_names, name)
        except Exception as e:
            log_line("Err", e)
            rmtree_hard(f"{repos_path}/{name}")
            assistant.pm.delete_uuids(
                value["uuid"] for value in actions_with_uuids.values()
            )

        return name

    def add_skill_from_local(self, path, assistant):
        self.ready_temp_dir()
        shutil.copytree(path, f"{repos_path}/temp")
        name = self.get_name()
        requirements = self.get_requirements()
        temp_name = self.rename_temp()
        requirements_names = self.install_requirements(assistant, requirements, name)
        prev_action_dict: dict = deepcopy(assistant.action_dict)
        try:
            self.rename_to_perm_name(name, temp_name)
        except:
            return False
        module = importlib.import_module(f"Skills.{name}")
        self.create_settings_meta(name)
        self.dump_meta_to_yaml(name)
        new_action_dict: dict = self.get_new_actions(
            assistant, prev_action_dict, new_action_dict
        )
        actions_with_uuids = assistant.pm.add_list(new_action_dict, name)
        self.insert_actions(actions_with_uuids, name)
        self.add_to_installed_skills(name)
        self.add_skill(assistant, name)
        self.add_requirements_to_db(requirements, requirements_names, name)

        return name

    def add_requirements_to_db(self, requirements, requirements_names, skill):
        for i in range(0, len(requirements)):
            url = requirements[i]
            name = requirements_names[i]

            con = sqlite3.connect("skills.db")

            cur = con.cursor()

            cur.execute(
                f"""
                    INSERT INTO requirements (url, name, requiredBy)
                    VALUES ('{url}', '{name}', '{skill}')
            """
            )

            con.commit()
            con.close()

    def is_folder_valid(self, folder_path):
        # Read the config file
        config_file_path = os.path.join(folder_path, "config.yaml")
        with open(config_file_path, "r") as config_file:
            config = yaml.safe_load(config_file)

        # Get the name from the config
        name = config.get("name")

        # Check if the name matches a file in the folder
        file_names = os.listdir(folder_path)
        if f"{name}.py" not in file_names:
            print("name not in file names")
            return False

        # Check if the folder contains a class with the same name
        module_file_path = os.path.join(folder_path, f"{name}.py")
        if not os.path.isfile(module_file_path):
            print("module file path not a file")
            return False

        # Read the file contents
        with open(module_file_path, "r") as module_file:
            file_contents = module_file.read()

        # Check if the class with the same name exists in the file
        if f"class {name}" not in file_contents:
            print("class name not in file contents")
            return False

        # Check if skill is already installed in the database
        conn = sqlite3.connect("skills.db")
        c = conn.cursor()
        value_to_check = name
        c.execute("SELECT COUNT(*) FROM actions WHERE skill = ?", (value_to_check,))
        result = c.fetchone()[0]
        conn.close()
        if result > 0:
            print("skill already installed")
            return False

        return True

    def remove_skill(self, skill_name: str, assistant: Assistant):
        print(skill_name)
        if not self.check_if_skill_is_needed(skill_name=skill_name):
            self.remove_from_memory(skill_name, assistant)
            self.remove_from_actions_table(skill_name)
            self.remove_from_installed_skills_table(skill_name)
            self.remove_from_vector_database(skill_name, assistant)
            self.delete_files(skill_name)
            dependancies = self.get_dependancies(skill_name)
            self.remove_from_requirements_table(skill_name)
            # self.remove_dependancies(dependancies, assistant)
            return {"status_code": 200}
        else:
            return {"status_code": 409}

    def get_dependancies(self, skill_name: str):
        con = sqlite3.connect("skills.db")
        cur = con.cursor()

        cur.execute("SELECT * FROM requirements WHERE requiredBy=?", (skill_name,))
        rows = cur.fetchall()

        con.close()

        return rows

    def remove_from_requirements_table(self, skill_name: str):
        con = sqlite3.connect("skills.db")
        cur = con.cursor()

        cur.execute("DELETE FROM requirements WHERE requiredBy=?", (skill_name,))

        con.commit()
        con.close()

    def remove_dependancies(self, dependancies, assistant):
        for row in dependancies:
            name = row[1]
            self.remove_skill(name, assistant)

    def delete_files(self, skill_name):
        rmtree_hard(os.path.join(repos_path, skill_name))

    def remove_from_memory(self, skill_name: str, assistant: Assistant):
        # reg.all
        reg_copy = reg.all.copy()
        for key, value in reg.all.items():
            if value["skill"].lower() == skill_name.lower():
                del reg_copy[key]

        reg.all = reg_copy

        # action_dict
        action_dict_copy = assistant.action_dict.copy()

        for key, value in assistant.action_dict.items():
            key: str
            skill = key.split("-")[0]

            if skill.lower() == skill_name.lower():
                del action_dict_copy[key]

        assistant.action_dict = action_dict_copy

        # installed_skills
        installed_skills_copy = assistant.installed_skills.copy()

        for key, value in assistant.installed_skills.items():
            key: str
            skill = key

            if skill.lower() == skill_name.lower():
                del installed_skills_copy[key]

        assistant.installed_skills = installed_skills_copy

    def remove_from_actions_table(self, skill_name: str):
        con = sqlite3.connect("skills.db")
        cur = con.cursor()

        cur.execute("DELETE FROM actions WHERE skill=?", (skill_name,))

        con.commit()
        con.close()

    def remove_from_installed_skills_table(self, skill_name: str):
        con = sqlite3.connect("skills.db")
        cur = con.cursor()

        cur.execute("DELETE FROM installedSkills WHERE skill=?", (skill_name,))

        con.commit()
        con.close()

    def remove_from_vector_database(self, skill_name: str, assistant: Assistant):
        assistant.pm.delete(
            {"operator": "Equal", "path": ["skill"], "valueText": skill_name}
        )

    def check_if_skill_is_needed(self, skill_name: str):
        con = sqlite3.connect("skills.db")
        cur = con.cursor()

        cur.execute("SELECT * FROM requirements WHERE name=?", (skill_name,))
        rows = cur.fetchall()
        con.close()

        if len(rows) > 0:
            return True
        else:
            return False

    def get_settings_meta(self, installed_skills):
        data = dict()
        for key, skill in installed_skills.items():
            path = os.path.join(repos_path, skill["name"], "settingsmeta.yaml")
            if os.path.exists(path):
                with open(path, "r") as stream:
                    data_loaded = yaml.safe_load(stream)
                    if (
                        data_loaded
                        and "skillMetadata" in data_loaded
                        and "sections" in data_loaded["skillMetadata"]
                    ):
                        updated_data = data_loaded.copy()
                        for section_index, section in enumerate(
                            data_loaded["skillMetadata"]["sections"]
                        ):
                            for field_index, field in enumerate(section["fields"]):
                                if "value" in field:
                                    if isinstance(field["value"], NoneType):
                                        updated_data["skillMetadata"]["sections"][
                                            section_index
                                        ]["fields"][field_index]["value"] = ""
                                else:
                                    updated_data["skillMetadata"]["sections"][
                                        section_index
                                    ]["fields"][field_index]["value"] = ""
                                if field["type"] == "select":
                                    options = field["options"].split(";")
                                    updated_data["skillMetadata"]["sections"][
                                        section_index
                                    ]["fields"][field_index]["options"] = {
                                        option.split("|")[0]: option.split("|")[1]
                                        for option in options
                                    }
                        else:
                            data_loaded = None
            else:
                data_loaded = None
            if data_loaded:
                data[skill["name"]] = data_loaded
        return data

    def remove_classes_and_functions(self, dictionary):
        # Iterate over the dictionary items
        dict_copy = deepcopy(dictionary)

        for key, value in dictionary.items():
            if isinstance(dict_copy[key], dict):
                dict_copy[key] = self.remove_classes_and_functions(value)
                if key in dict_copy and "actions" in dict_copy[key]:
                    dict_copy[key]["actions"] = list(dictionary[key]["actions"].keys())
            elif callable(value) or inspect.isclass(value):
                # Remove classes and functions
                del dict_copy[key]
            else:
                try:
                    json.dumps(value)
                except (TypeError, OverflowError):
                    del dict_copy[key]
        return dict_copy

    def get_installed_skills(self, assistant: Assistant):
        return self.remove_classes_and_functions(assistant.installed_skills)

    def get_config(self, skill_name):
        path = os.path.join(repos_path, skill_name, "settings.yaml")
        if os.path.exists(path):
            with open(path) as file:
                documents = yaml.full_load(file)
                return documents

    def get_setting(self, skill_name, field_name):
        config = self.get_config(skill_name)
        return config[field_name]

    def get_settings_meta_for_skill(self, skill_name, assistant):
        settings = self.get_settings_meta(assistant.installed_skills)[skill_name]
        for section in settings["skillMetadata"]["sections"]:
            for field in section["fields"]:
                if "name" in field:
                    value = self.get_setting(skill_name, field["name"])
                    field["value"] = value
        return settings
