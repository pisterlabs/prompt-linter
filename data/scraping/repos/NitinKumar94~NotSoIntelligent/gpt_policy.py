from __future__ import annotations

import abc
import copy
import logging
from enum import Enum
from pathlib import Path
from rasa.shared.core.events import Event, SlotSet
from typing import (
    Any,
    List,
    Optional,
    Text,
    Dict,
    Callable,
    Tuple,
    TypeVar,
    TYPE_CHECKING,
)

import numpy as np

from rasa.engine.graph import GraphComponent, ExecutionContext
from rasa.engine.storage.resource import Resource
from rasa.engine.storage.storage import ModelStorage
from rasa.core.featurizers.precomputation import MessageContainerForCoreFeaturization
from rasa.core.featurizers.tracker_featurizers import TrackerFeaturizer
from rasa.core.featurizers.tracker_featurizers import MaxHistoryTrackerFeaturizer
from rasa.core.featurizers.single_state_featurizer import SingleStateFeaturizer
from rasa.core.featurizers.tracker_featurizers import FEATURIZER_FILE
import rasa.utils.common
import rasa.shared.utils.io
from rasa.shared.exceptions import RasaException, FileIOException
from rasa.shared.nlu.constants import ENTITIES, INTENT, TEXT, ACTION_TEXT, ACTION_NAME
from rasa.shared.core.domain import Domain, State
from rasa.shared.core.trackers import DialogueStateTracker
from rasa.shared.core.generator import TrackerWithCachedStates
from rasa.core.constants import (
    DEFAULT_POLICY_PRIORITY,
    POLICY_PRIORITY,
    POLICY_MAX_HISTORY,
    DEFAULT_MAX_HISTORY
)
from rasa.shared.core.constants import USER, SLOTS, PREVIOUS_ACTION, ACTIVE_LOOP
import rasa.shared.utils.common
from rasa.engine.recipes.default_recipe import DefaultV1Recipe
from rasa.core.policies.policy import Policy, PolicyPrediction
from langchain.chat_models import ChatOpenAI
import json, os

if TYPE_CHECKING:
    from rasa.shared.nlu.training_data.features import Features


logger = logging.getLogger(__name__)


@DefaultV1Recipe.register(
    [DefaultV1Recipe.ComponentType.POLICY_WITHOUT_END_TO_END_SUPPORT], is_trainable=True
)
class GPTPolicy(Policy):
    def __init__(
        self,
        config: Dict[Text, Any],
        model_storage: ModelStorage,
        resource: Resource,
        execution_context: ExecutionContext,
        featurizer: Optional[TrackerFeaturizer] = None,
    ) -> None:
        """Initializes the policy."""

        super().__init__(
            config, model_storage, resource, execution_context, featurizer
        )
        self.api_key = os.environ['OPENAI_API_KEY']
        self.chat = ChatOpenAI(temperature=0.3, openai_api_key=self.api_key)

    @staticmethod
    def get_default_config() -> Dict[Text, Any]:

        return {
            POLICY_MAX_HISTORY: DEFAULT_MAX_HISTORY,
            POLICY_PRIORITY: DEFAULT_POLICY_PRIORITY
        }

    def train(
        self,
        training_trackers: List[TrackerWithCachedStates],
        domain: Domain,
        **kwargs: Any,
    ) -> Resource:

        self.persist()
        return self._resource
    
    def predict_action_probabilities(
        self,
        tracker: DialogueStateTracker,
        domain: Domain,
        rule_only_data: Optional[Dict[Text, Any]] = None,
        **kwargs: Any,
    ) -> PolicyPrediction:

        states = tracker.past_states(domain=domain)
        
        logger.debug(f"""Tracker states:{self.format_tracker_states(states)}""")

        action_probabilities = self._default_predictions(domain=domain)
        upstream_intent = tracker.latest_message.intent
        next_action = 'llm_call'
        action_index= domain.index_for_action(next_action)
        action_probabilities[action_index] = 1.0
        
        for idx in range(len(domain.action_names_or_texts)):
            logger.debug(f"{idx}: {domain.action_names_or_texts[idx]}, prob {action_probabilities[idx]}")

        predictions =  PolicyPrediction(
            probabilities=action_probabilities,
            policy_name=self.__class__.__name__,
            policy_priority=1,
            events=None,
            optional_events=None,
            is_end_to_end_prediction=False,
            is_no_user_prediction=False,
            diagnostic_data=None,
            hide_rule_turn=False,
            action_metadata=None
        )

        return predictions
    
    @classmethod
    def _metadata_filename(cls) -> Text:
        return "gpt_policy.json"

    def persist(self) -> None:
        """Persists the policy to storage."""
        with self._model_storage.write_to(self._resource) as path:
            
            file = Path(path) / self._metadata_filename()

            rasa.shared.utils.io.create_directory_for_file(file)
            rasa.shared.utils.io.dump_obj_as_json_to_file(file, {"gpt_policy": "dummy_data"})

    