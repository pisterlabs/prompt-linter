import os
import re

import logging
from typing import Dict, List, Tuple

from jarvis.const import ACTION_CHAIN_PATH, COMMON_ACTION_CHAINS
from jarvis.actions import ActionBase, ActionChain, action_registry
from jarvis.actions.context import Context
from jarvis.automation.browser import create_browser_automation
from jarvis.automation.desktop import create_desktop_automation
from jarvis.nlp import nlp_utils


class PhraseMatcher:
    def __init__(self, phrase, conjunctive=False):
        super()

        self._phrase = phrase
        self._matcher = self._convert_to_regex(phrase)
        self._conjunctive = conjunctive

    def match(self, cmd) -> Tuple[bool, dict]:
        """Returns a tuple of (success, matched_params_dict)"""
        m = self._matcher.match(cmd)
        if m is None: return (False, None)

        return (True, m.groupdict())

    def is_conjunctive(self):
        return self._conjunctive

    def _convert_to_regex(self, phrase):
        # TODO(hari): Support and handle different types
        # of variables here. For now everything is assumed
        # as a string.
        regexp = phrase.replace('{', '(?P<').replace('}', '>.+)')
        # logging.info(phrase)
        # logging.info(regexp)
        return re.compile(regexp)


class ActionResolver:
    """The "Brain" which routes commands to actions.

    Inputs include the transcribed command, current application state (e.g. open windows,
    previous command), and the library of supported actions. These inputs are combined to
    determine which Action to perform.
    """

    # Some files are platform specific like mac_automation.py.
    # To avoid loading them, we only load files that contain actions
    # which are determined by looking for file with this suffix.
    ACTIONS_FILE_SUFFIX = "_actions.py"

    def __init__(self):
        super()

        self.desktop = create_desktop_automation()
        self.context = Context(desktop=self.desktop)
        self._browser = None

        # Map between phrase string and the action type
        self._phrase_map: Dict[PhraseMatcher, ActionBase] = {}
        # Support multiple actions by using conjunctive and.
        # This will not have any action mapped to it.

        # TODO(bfortuner): Support multiple actions
        # Right now this makes the resolve logic complex and breaks
        # when a user says the word "and" as part of a link name
        # self._phrase_map[PhraseMatcher("{a} and {b}", True)] = None

        # Loads Actions from Classes
        self._phrase_map.update(self._find_action_phrases("jarvis/automation"))

        # Loads community action chains
        self._phrase_map.update(self._load_action_chain_phrases(
            COMMON_ACTION_CHAINS)
        )
        # Load user's local action chains
        self._user_action_chains = []
        self.reload_user_action_chains()

        logging.info(f"Found {len(self._phrase_map)} phrases to match against")
        self._action_classes = action_registry.load_action_classes_from_modules("jarvis/automation")

    @property
    def browser(self):
        if self._browser is None:
            self._browser = create_browser_automation(self.desktop)
        return self._browser

    def parse(self, cmd: str) -> List[ActionBase]:
        """Converts user command string in list of Actions.

        NOTE: Right now we only support 1 action per command, but we
        return a list of matching actions to attempt ordered by the
        most relevant as determined by context (e.g. active window).
        """

        logging.info(f"CommandParser Input:  {cmd}")
        cmd = nlp_utils.normalize_text(cmd)
        logging.info(f"Command after cleaning: {cmd}")

        # This can contain action chain instances
        matching_actions = self._find_matching_actions(cmd)

        if len(matching_actions) == 0:
            if not os.getenv("NO_OPENAI") and os.getenv("OPENAI_API_KEY"):
                return self.attempt_model_based_resolve(cmd)
            else:
                logging.info("No command matched with that!")
                raise NotImplementedError("No command matched with that!")

        # Use Context to disambiguate commands
        actions = self._sort_actions_by_relevance(matching_actions)

        # Initialize the actions
        action_instances = []
        for action_cls, action_params in actions:
            # TODO: Refactor all of this. There is a much better abstraction
            # which unifies Action and ActionChain, I just haven't found it yet.
            if isinstance(action_cls, ActionChain):
                action_cls.add_automations(self.desktop, self.browser)
                action_instances.append(action_cls)
            else:
                if "browser" in action_cls.automations():
                    action_params["browser"] = self.browser
                if "desktop" in action_cls.automations():
                    action_params["desktop"] = self.desktop
                action_instances.append(action_cls(**action_params))
        return action_instances

    def attempt_model_based_resolve(self, cmd: str) -> List[ActionChain]:
        # NOTE: This requires an Open.ai API key
        print(f"Attemping model-based resolve for command: '{cmd}'")
        from jarvis.nlp.openai import openai_action_resolver
        chain = openai_action_resolver.infer_action_chain(cmd, self._action_classes)
        chain.add_automations(self.desktop, self.browser)
        return [chain]

    def _sort_actions_by_relevance(self, matching_actions):
        # If the active window supports the action, push it to the top
        ordered_actions = []
        active_window_actions = []
        for i, (action_type, _) in enumerate(matching_actions):
            if self.context.active_window in action_type.app_names():
                active_window_actions.append(matching_actions[i])
            else:
                ordered_actions.append(matching_actions[i])
        return active_window_actions + ordered_actions

    def _find_matching_actions(self, phrase: str):
        # Returns list of (action type, action params)
        # HACK: We always reload the user action phrases so we can pickup
        # updates for newly created action chains
        self._phrase_map.update(self._load_action_chain_phrases(
            self._user_action_chains)
        )
        actions = []
        for phrase_matcher, action_type in self._phrase_map.items():
            status, params = phrase_matcher.match(phrase)
            if status:
                actions.append((action_type, params))
        return actions

    def _find_action_phrases(self, dir_name: str) -> dict:
        """Loop through sub directories and load all the packages
        and return them as a dictionary"""
        phrase_map = {}
        file_list = os.listdir(os.path.join(os.getcwd(), dir_name))
        for file_name in file_list:
            full_path = os.path.join(os.path.abspath(dir_name), file_name)
            rel_path = os.path.join(dir_name, file_name)
            if os.path.isdir(full_path) and \
                os.path.exists(os.path.join(full_path, "__init__.py")):
                phrase_map.update(self._find_action_phrases(rel_path))
            elif full_path.endswith(self.ACTIONS_FILE_SUFFIX) and file_name != "__init__.py":
                module_name = os.path.splitext(file_name)[0]
                logging.info(module_name)
                module = __import__(
                    f"{dir_name.replace(os.sep, '.')}.{module_name}",
                    fromlist = ["*"])
                for _, t_value in module.__dict__.items():
                    try:
                        if issubclass(t_value, ActionBase):
                            for phrase in t_value.phrases():
                                # TODO(hari): Validate the phrases to make sure
                                # they don't use invalid param names and types
                                phrase_matcher = PhraseMatcher(phrase)
                                phrase_map[phrase_matcher] = t_value
                    except:
                        # Some members of the module aren't classes. Ignore them.
                        pass
        return phrase_map

    def _load_action_chain_phrases(self, chains: List[Dict]):
        # TODO: Allow users to register custom action chains
        phrase_map = {}
        for chain_dict in chains:
            chain = ActionChain.from_dict(chain_dict)
            for phrase in chain.phrases:
                matcher = PhraseMatcher(phrase)
                phrase_map[matcher] = chain
        return phrase_map

    def reload_user_action_chains(self, fpath: str = ACTION_CHAIN_PATH):
        self._user_action_chains = action_registry.load_action_chains_from_file(fpath)
