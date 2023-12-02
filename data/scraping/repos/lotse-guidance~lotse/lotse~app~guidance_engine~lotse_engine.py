import logging
import os.path
from typing import List

from rickled import ObjectRickler, Rickle

from guidance_strategies.action import ConditionalGuidanceAction
from guidance_strategies.strategy import Strategy
from guidance_strategies.suggestion import SuggestionModel
from guidance_strategies.meta_strategy import MetaStrategy


class ContextVector:
    def initialize(self):
        pass


class LotseEngine:
    logger = logging.getLogger(__name__)

    def __init__(self, strategy_path: str, state_path: str, meta: str):
        """

        :param strategy_path: The path from which to read the strategy config files
        """
        self.strategies: List[Strategy] = []
        self.meta_strategy = MetaStrategy()
        self.applicable_strategies = []
        rickler = ObjectRickler()

        files = [file for file in os.listdir(strategy_path) if
                 os.path.isfile(os.path.join(strategy_path, file)) and file.endswith('.yaml') and not file == 'meta.yaml']
        self.logger.debug(f"Loading files {files}")
        for file in files:
            self.logger.debug(f"Loading file {file}")
            rickle = Rickle(file, deep=True, load_lambda=True, path=strategy_path)
            strat = rickler.from_rickle(rickle, Strategy)
            self.strategies.append(strat)
        self.logger.info(f"Guidance engine initialized {len(self.strategies)} strategies.")

        meta = os.path.join(strategy_path, 'meta.yaml')
        if os.path.isfile(meta):
            rickle = Rickle(meta, deep=True, load_lambda=True, path=strategy_path)
            self.meta_strategy = rickler.from_rickle(rickle, MetaStrategy)
            print("Successfully loaded meta strategy.")

        state = os.path.join(state_path, 'vector.yaml')
        state_vector_rick = Rickle(state, deep=True, load_lambda=True, path=state_path)
        state_vector: ContextVector = rickler.from_rickle(state_vector_rick, ContextVector)
        state_vector.initialize()
        print(state_vector.__dict__)

        self.current_state = state_vector
        self.last_delta = None
        self.conditional_actions: List[ConditionalGuidanceAction] = []
        self.suggestions: List[SuggestionModel] = []

    def get_applicable_strategies(self) -> List[Strategy]:
        return list(filter(lambda s: s.determine_applicability(self.current_state, self.last_delta), self.strategies))

    def generate_conditional_actions(self):
        self.conditional_actions = []
        for strategy in self.applicable_strategies:
            for action in strategy.generate_actions():
                self.conditional_actions.append(action)
        return self.conditional_actions

    def get_applicable_actions(self):
        actions = []
        for action in self.conditional_actions:
            if action.is_applicable(self.current_state, self.last_delta):
                actions.append(action)
                action.suggested = True
        return actions

    def generate_suggestions(self) -> List[SuggestionModel]:
        actions = self.get_applicable_actions()
        print(f'got {len(actions)} actions to apply in the current context')
        if len(actions) > 0:
            actions = self.meta_strategy.filter_actions(actions, self.current_state)
            print(f'got {len(actions)} actions remain after meta strategy filtering.')
        new_suggestions = [action.generate_suggestions(self.current_state) for action in actions]
        new_suggestions = list(filter(lambda s: s is not None, new_suggestions))
        print(f'obtained {len(new_suggestions)} new suggestions')
        self.suggestions.extend(new_suggestions)
        print(f'total suggestions: {len(self.suggestions)}')
        return new_suggestions

    def suggestions_to_retract(self) -> List[SuggestionModel]:
        return list(filter(lambda s: s.action.should_retract(self.current_state, self.last_delta, s), self.suggestions))
