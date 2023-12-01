#!/usr/bin/python3
import subprocess
import openai
import json
import yaml

from utils import ROBOT_DETAILS, SPAWN_TERMINAL, YAML_DICT_TEMPLATE, jarvis_say, find_file

openai.api_key_path = ".env"
MODEL = "gpt-3.5-turbo"

def launch_rviz():
    jarvis_say("Launching RViz")
    return subprocess.Popen(
        f"{SPAWN_TERMINAL} ros2 launch crazyflie launch.py",
        shell=True
    )

def launch_cfclient():
    jarvis_say("Launching CFClient")
    return subprocess.Popen(
        f"{SPAWN_TERMINAL} cfclient",
        shell=True
    )

def run_crazyflie_script(file_name: "str"):
    package = find_file(file_name)
    if package is None: return jarvis_say(f"Unable to find: {file_name}")

    cmd = f"ros2 run {package} {file_name[:-3]}"
    jarvis_say(f"Running {file_name} from {package}")
    return subprocess.Popen(
        f"{SPAWN_TERMINAL} bash -c 'echo {cmd}; {cmd}; read -p \"Press enter to continue\" line'",
        shell=True
        )

def generate_crazyflie_yamlfile(drones):
    jarvis_say("Generating crazyflie YAML file")
    enabled_drones = {}
    for drone in drones:
        drone_id = f"cf{drone['drone_number']}"
        initial_position = [drone["initial_position_x"], drone["initial_position_y"], drone["initial_position_z"]]
        jarvis_say(f"Enabling drone {drone_id} at position {initial_position}")
        enabled_drones[drone_id] = {
            "enabled": True,
            "initial_position": initial_position,
            **ROBOT_DETAILS[drone_id]
        }
    config = YAML_DICT_TEMPLATE.copy()
    config["robots"] = enabled_drones
    with open("crazyflie.yaml", "w") as f:
        yaml.dump(config, f, default_flow_style=None)
    return jarvis_say("Writing crazyflie.yaml file")
    

functions = {
    "launch_rviz": launch_rviz,
    "launch_cfclient": launch_cfclient,
    "run_crazyflie_script": run_crazyflie_script,
    "generate_crazyflie_yaml": generate_crazyflie_yamlfile
}

if __name__ == "__main__":
    while (input_string := input("JARVIS> ")) != "quit":
        response = openai.ChatCompletion.create(
            model=MODEL,
            messages=[
                {"role": "user", "content": input_string},
            ],
            functions=[
                {
                    "name": "launch_rviz",
                    "description": "Launches the RViz software",
                    "parameters": {
                        "type": "object",
                        "properties": {}
                    }
                },
                {
                    "name": "launch_cfclient",
                    "description": "Launches the CFClient software",
                    "parameters": {
                        "type": "object",
                        "properties": {}
                    }
                },
                {
                    "name": "run_crazyflie_script",
                    "description": "Runs a crazyflie demo script in RViz. The file name must always end in .py",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "file_name": {
                                "type": "string"
                            }
                        }
                    }
                },
                {
                    "name": "generate_crazyflie_yaml",
                    "description": "Generates a crazyflie YAML file for configuring which crazyflie are enabled and their initial positions",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "drones": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "drone_number": {
                                            "type": "number",
                                        },
                                        "initial_position_x": {
                                            "type": "number"
                                        },
                                        "initial_position_y": {
                                            "type": "number"
                                        }, 
                                        "initial_position_z": {
                                            "type": "number"
                                        }
                                    }
                                }
                            }
                        }
                    }
                },
            ],
            function_call="auto",
            temperature=0,
        )

        response = response["choices"][0]["message"]  # type: ignore
        if response.get("function_call"):  # type: ignore
            called_function = functions[response["function_call"]["name"]]
            called_function(**json.loads(response["function_call"]["arguments"]))