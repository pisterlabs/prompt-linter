# Kube-chat powered by OpenAI's chat GPT
# First layer: split user intent into actions
# Second layer, if transform/apply action: fine grain target resources
# Third layer, if transform/apply action: generate code to transform resources
# Second layer if simple chat: talk with users

import asyncio
import ast
import yaml
import openai
import re
import logging
import json
import time
import random
import os
import sys
from prompt_toolkit import prompt
from prompt_toolkit.clipboard.pyperclip import PyperclipClipboard
from prompt_toolkit.completion import WordCompleter, FuzzyWordCompleter
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
from prompt_toolkit.history import FileHistory
from pygments import highlight
from pygments.lexers import YamlLexer
from pygments.formatters import Terminal256Formatter

from commands.base_commands import KubeCommand
from commands.simple_commands import ChatCommand
from commands.transform_commands import TransformCommand
from commands.actions_commands import ActionCommand

from kubernetes import client, config
from typing import List, Dict
from collections import defaultdict

commands = [KubeCommand, ChatCommand, TransformCommand, ActionCommand]

log_dir = os.environ.get("ANSIBLE_CHAT_LOG_DIR", "./logs")
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
logging.basicConfig(filename=os.path.join(log_dir, "debug.log"), level=logging.DEBUG)

class KubeChat:
    # keep user / assistant first prompts layer
    messages_list = []
    # Save messages dir (for later reload when implemented)
    messages_dir = os.environ.get("ANSIBLE_CHAT_MESSAGES_DIR", "./messages")

    # workspace memory of all items created by bot
    my_items = []

    def __init__(self, command_classes):
        messages_file = os.path.join(self.messages_dir, "messages.json")

        self.command_instances = {}

        self.command_mapping = {
            command_class.command: {
                "class": command_class,
                "command": command_class.command,
                "description": command_class.description,
                "parameters": command_class.parameters,
            }
            for command_class in command_classes
        }
        # Generate singletons
        for i in self.command_mapping.keys():
            if i == "":
                self.command_instances["/kube_command"] = self.command_mapping[i]["class"]()
            else:
                self.command_instances[i] = self.command_mapping[i]["class"]()

        # self.load_files()
        self.load_messages()

    async def chat(self, input):
        # important ! Injected in do_action since intent is no good at injectting prompt :/
        self.user_input = input
        command_name, user_params, text = self.parse_input(user_input)
        response = await self.handle_command(command_name, user_params, text)

        # TODO: keep only two last messages for do_action self.my_items to avoid >4k

    # TODO: explain this black magic function !
    def prepare_action(
        self, action, **kwargs
    ):
        intent = list(action.keys())[0]
        action[intent].pop('assistant_follow', "")

       # Explain actions 
        original_user=action[intent].get("original_user", "")
        if "original_user" in action.keys():
            original_user+=action['original_user'] 
        if not original_user == "":
            action[intent]['user'] = original_user

        answer = self.command_instances["/action"].new_message(
            f"{action}", self.my_items
            )
        try:
            action = ast.literal_eval(answer.gpt_answer)
        except:
            print (answer.gpt_answer)
        return action


    # This is second layer prompt, as planned by actions served from first /chat command
    # redirects to accurate subprompt and transform yam or act on cluster, or simply discuss
    # also try to extract items from action list
    def do_action(
        self, action, **kwargs
    ):
        intent = list(action.keys())[0]
        values = action[intent]

        # FIXME: remove duplicates already existing that can sometimes appear again ?
        to_create_items_list=values.get("to_create_list", [])+values.get("to_add_list", [])+values.get("to_create_and_patch_list", [])
        if "to_create_list" in action.keys():
            to_create_items_list+=action['to_create_list']

        to_change_items_list=values.get("to_patch_list", []) + values.get("to_update_list", [])
        if "to_update_list" in action.keys():
            to_change_items_list+=action['to_update_list'] 

        to_delete__items_list=values.get("to_delete_list", [])
        if "to_delete_list" in action.keys():
            to_delete__items_list+=action['to_delete_list'] 


        related_items_list=values.get("target_list", [])
        if "target_list" in action.keys():
            related_items_list+=action['target_list'] 
        
        action_target_existing_items= to_change_items_list + related_items_list
        current_items = self.my_items.copy()

        prompt_extract = values.get("user", "")
        if intent in ["create", "change", "patch", "add", "deploy", "change", "create_and_patch", "update", "create_and_update"]:
            # TODO: try catch error
            answer = self.command_instances["/transform"].new_message(
                intent, prompt_extract,
                self.extract_items(self.my_items, action_target_existing_items + to_change_items_list + related_items_list), 
                to_create_items_list,
                to_change_items_list
            )
            self.my_items=self.merge_items(answer.items,self.my_items)

            print(highlight(yaml.dump(self.my_items), YamlLexer(), Terminal256Formatter())) #material

            # FIXME: make pdb/hpa reco once only
            pdbs = [item for item in self.my_items if item.get('kind') == 'PodDisruptionBudget']
            # Also hpa, and propose only if new Dp with no pdb
            if len(pdbs)==0:
                try: 
                    actions = ast.literal_eval(self.command_instances["/kube_command"].conversation_messages[-1]['content'])
                    self.command_instances["/kube_command"].conversation_messages[-1]['content'] = str(actions)
                except:
                    pass
        elif intent in ["status"]:
            self.print_slow("""KubeChat: Sure! This would be a pleasure to gather information on cluster and give you a human readable status. Something like: "The nginx pod is started but I can see there's a problem with readyness probe". If only you worked harder on this implementation instead of watching series like The Last Of Us, you could have make a great demo but...""")
        elif intent in ["delete"]:
            return current_items
        elif intent in ["apply"]:
            from subprocess import Popen, PIPE, STDOUT
            import re

            for p in self.my_items:
                logging.debug(f"applying {p}")
                doc="---\n"+ yaml.dump(p)
                p = Popen(['kubectl', 'apply', '-f', '-'],  stdin=PIPE, stdout=PIPE)
                stdout_data = p.communicate(input=doc.encode())[0]
                # remove if any newline in stdout_data
                logging.debug(f"applyied {p}: {stdout_data}")
                if  stdout_data is not None:
                    stdout_data_no_eol = re.sub(r"\n", "", stdout_data.decode())
                    stdout_data = re.sub(r"\r", "", stdout_data_no_eol)
                    self.print_slow( stdout_data_no_eol + " ✅\n")

        elif intent in ["XXXcommand", "XXXlogs", "XXXkill"]:  # FIXME: 'troubleshoot'
            pass
        elif intent == "explain":
            answer = self.command_instances["/chat"].new_message(
                prompt_extract, self.extract_items(self.my_items, action_target_existing_items + to_change_items_list + related_items_list,remove_extracted=False)
            )
        else:
            answer = self.command_instances["/chat"].new_message(
                prompt_extract, self.extract_items(self.my_items, action_target_existing_items + to_change_items_list + related_items_list,remove_extracted=False)
            )

            pass
        # FIXME : may not be needed => one shot prompt! 
        self.command_instances["/transform"].conversation_messages = self.command_instances["/transform"].conversation_messages[-2:]
        # FIXME: chat shoudl better route specific one shot when explaining aciting items (leads to big payloads)
        self.command_instances["/chat"].conversation_messages = self.command_instances["/chat"].conversation_messages[-2:]
        self.command_instances["/kube_command"].conversation_messages = self.command_instances["/kube_command"].conversation_messages[-8:]
        return current_items

    def parse_input(self, user_input):
        command_name = None
        user_params = {}
        text = ""

        parts = user_input.split(maxsplit=1)
        if parts and parts[0].startswith("/"):
            command_name = parts[0]
            if len(parts) > 1:
                remaining_text = parts[1]

                # Match parameters with the pattern: key:'value'
                pattern = re.compile(r"(\w+):'([^']+)'")
                matches = pattern.finditer(remaining_text)

                for match in matches:
                    key, value = match.groups()
                    user_params[key] = value

                # Remove matched parameters and leading/trailing whitespaces from the text
                text = pattern.sub("", remaining_text).strip()
        else:
            text = user_input

        if command_name == None:
            command_name = "/kube_command"
        return command_name, user_params, text

    async def handle_command(self, command_name, user_params, text):
        if command_name is None:
            command_class = KubeCommand
            command_name = "kube_command"  # Add a name for the default  KubeCommand
        else:
            command_data = self.command_mapping.get(
                command_name, {"class": KubeCommand}
            )
            command_class = command_data["class"]
        
        self.print_slow("...thinking...\n", 6000)
        command_instance = self.command_instances[command_name]
        answer = command_instance.new_message(text, self.my_items, **user_params)
        current_items = []

        # launch do_action from intent
        # force definitions for some do_action call
        target_items=self.my_items.copy()
        # TODO: try catch fallback prompt dindt understand
        actions = ast.literal_eval(answer.gpt_answer)
        
        if type(actions) is list:
            for action in actions:

                action = self.prepare_action(action)
                # FIXME :  chat or explain may rephrase problem => rejinject original prompt
                self.do_action(action)
        else:
            action = self.prepare_action(actions)
            self.do_action(action)
       # FIXME: need better exception handling
        # except Exception as e:
        #     logging.debug(e)
        #     self.do_action({'explain': {'user': f"something went wrong with this prompt. Probably ive done something wrong.\nPrompt: {text}"}})

        return answer.gpt_answer

    def add_conversation_messages(self, messages):
        self.messages_list.extend(messages)
        self.save_messages()

    def load_messages(self):
        try:
            with open(self.messages_file, "r") as f:
                self.messages_list = json.load(f)
        except Exception:
            self.messages_list = []

    def save_messages(self):
        with open(self.messages_file, "w") as f:
            json.dump(self.messages_list, f, indent=2)

    def merge_items(self,new_items,items):
        logging.debug(f'Merging {new_items} with {items}')

        for saved_item in reversed(items):
            found=False
            for current_item in new_items:
                if current_item['kind'] == saved_item['kind'] and current_item['metadata']['name'] == saved_item['metadata']['name'] \
                and ('namespace' not in current_item['metadata'] or ('namespace' in saved_item and current_item['metadata']['namespace'] == saved_item['metadata']['namespace'])):
                    logging.debug(f"D already found: {saved_item}")
                    found=True
                    break
            if not found:
                new_items.insert(0,saved_item)
        logging.debug(f'Merge result {new_items}')
        return new_items

    # TODO: load and save files
    def extract_items(self,current_items, action_target_existing_items, remove_extracted=True):
        extracted_items = []
        for new_item in action_target_existing_items:
            for current_item in current_items:
                if current_item['kind'] == new_item['kind'] \
                and ('name' in current_item['metadata'] and current_item['metadata']['name'] == new_item['name']):
                # and ('namespace' not in current_item['metadata'] or ('namespace' in new_item and current_item['metadata']['namespace'] == new_item['namespace']))
                    extracted_items.append(current_item)
                    if remove_extracted:
                        current_items.remove(current_item)
                    break
        return extracted_items

    # FIXME: flatten is defined twice !!!
    def flatten_items(self, item_list):
        target_items = ""
        for item in item_list:
            metadata = item['metadata']
            target_items += f"{item['kind']} {metadata['name']}"
            if 'namespace' in metadata: target_items += f" in namespace {metadata['namespace']}"
            target_items += f","
        return target_items

    def print_slow(self, str, typing_speed=600):
        for letter in str:
            print(letter, end="")
            time.sleep(random.random()*10.0/typing_speed)


def load_kube_config():
    try:
        config.load_kube_config()
    except config.ConfigException:
        config.load_incluster_config()

def sort_conditions(conditions: List[dict]) -> List[dict]:
    return sorted(conditions, key=lambda x: x["last_transition_time"])

def get_pods_by_label(labels: Dict[str, str], namespace="default") -> List[str]:
    api_instance = client.CoreV1Api()
    label_selector = ','.join([f'{key}={value}' for key, value in labels.items()])
    try:
        # TODO: support per namespace filter
        api_response = api_instance.list_pod_for_all_namespaces(label_selector=label_selector)

        return [pod.metadata.name for pod in api_response.items]
    except client.exceptions.ApiException as e:
        print(f"Error calling Kubernetes API: {e}")
        return []

# TODO: would diserve to to be really implemented (scope TBD)
def get_status_for_pods(pod_list: List[str], namespace="default") -> List[Dict]:
    core_instance = client.CoreV1Api()
    apps_instance = client.AppsV1Api()
    status = ""

    print("***")
    
    statuses = []
    for pod_name in pod_list:
        
        try:
            pod_status = core_instance.read_namespaced_pod_status(pod_name, namespace)

            conditions = pod_status.status.conditions

            # pod.status.container_statuses[0].state.terminated
            sorted_conditions = sorted(conditions, key=lambda x: x.last_transition_time)
            
            # Keep only two latest of each type
            filtered_conditions = []
            type_counter = defaultdict(int)

            for condition in sorted_conditions:
                condition_type = condition.type
                if type_counter[condition_type] < 2:
                    filtered_conditions.append(condition)
                    type_counter[condition_type] += 1
            condtions_text = ""
            for i in filtered_conditions:
                condtions_text += f"\n- last transition time: {i.last_transition_time}, "
                condtions_text += f"{i.type} is {i.status}"
                if i.reason:
                    condtions_text += f" because {i.reason}."
                if i.message:
                    condtions_text += f"Additional info: {i.message}"

            status = f"{pod_name}" + condtions_text
            statuses += [status] 
            dp_status = apps_instance.read_namespaced_deployment_status(name="gloups", namespace="test-gloups")

        except client.exceptions.ApiException as e:
            print(f"Error calling Kubernetes API: {e}")

    return statuses


if __name__ == "__main__":
    try:
        openai.api_key = os.environ["OPENAI_API_KEY"]
    except: 
        print("Please run command `export OPENAI_API_KEY=xxx-your-api-key` before launching Kubechat. Create one here: https://platform.openai.com/account/api-keys.")
        sys.exit(-1)
    kube_chat = KubeChat(commands)
    kube_chat.my_items = [ ]
    print(
        "Welcome to Kubernetes chat powered by GPT-3 ! Type '/exit' to end the chat."
    )
    print("Press *** ALT+Enter *** to validate a message, simple enter will only go next line.")
    print("/exit to exit")

    from prompt_toolkit import PromptSession
    home = os.path.expanduser("~")
    history_file = os.path.join(home, '.kube-chat.history')

    session = PromptSession(history=FileHistory(history_file))
    while True:
        try:
            command_completer = FuzzyWordCompleter(['create', 'delete', 'update', 'apply', 'patch', 'get', 'node', 'pod', 'container', 'HorizontalPodAutoscaler', 'PodDisruptionBudget', 'replicaset', 'deployment', 'statefulset', 'daemonset', 'job', 'cronjob', 'resource', 'quota', 'limitrange', 'persistentvolume', 'persistentvolumeclaim', 'storageclass', 'volume', 'configmap', 'secret', 'service', 'ingress', 'endpoint', 'loadbalancer', 'networkpolicy', 'role', 'rolebinding', 'clusterrole', 'clusterrolebinding', 'serviceaccount', 'token', 'certificate', 'apiserver', 'etcd', 'kubelet', 'kube-proxy', 'kube-controller-manager', 'kube-scheduler', 'coredns', 'fluentd', 'envoy', 'istio', 'calico', 'cilium', 'prometheus', 'grafana', 'jaeger', 'linkerd', 'rancher', 'rook', 'traefik', 'argocd'])
            user_input = session.prompt(
                "You: ",
                multiline=True,
                mouse_support=False,
                clipboard=PyperclipClipboard(),
                completer = command_completer,
                complete_while_typing=False,
                auto_suggest=AutoSuggestFromHistory()
            )

            if user_input != "":
                if user_input.lower().startswith("/exit"):
                    exit(0)
                kube_chat.user_full_input = user_input
                asyncio.run(kube_chat.chat(user_input))
                print("")
        except KeyboardInterrupt:
            pass

