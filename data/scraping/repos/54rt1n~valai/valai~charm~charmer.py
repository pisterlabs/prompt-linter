# valai/charm/charmer.py

from collections import defaultdict
import logging
import os
from typing import List, Optional

from .shadow import ContextShadowing
from .prompt import Librarian
from .guidance import GuidanceStrategy

logger = logging.getLogger(__name__)


class Charmer:
    """The Charmer is the main interface to the Charm system. It is responsible for
    managing the game state, and the interaction between the player and the system.
    """
    def __init__(self, library : Librarian, shadow : ContextShadowing, guidance : GuidanceStrategy, history_threshold : int = 100) -> None:
        self.library = library
        self.shadow = shadow
        self.turn_count = 0
        self.recent = {}
        self.history_threshold = history_threshold
        self.past_history = []
        self.current_history = []
        self.char_dialog = defaultdict(list)
        self.guidance = guidance

    def init_history(self, load : bool = False, **kwargs) -> bool:
        history = None
        if load: 
            history = self.load_game_text()
        initial_q = "> New Game"
        initial_a = "Narrator: (informative) Welcome to Verana"
        if history is None:
            history = [initial_q, initial_a]
        self.past_history = []
        self.current_history = history
        return self.roll_history(**kwargs)

    def roll_history(self, force_roll : bool = False, **kwargs) -> bool:
        # we want to preserve # Codex
        if len(self.current_history) > self.history_threshold or force_roll:
            # cut current history in half
            ix = len(self.current_history)//2
            codex = [e for e in self.current_history if e[:5] == 'Codex']
            # TODO codex (codices should be preserved, just rolled to the beginning)
            self.past_history += self.current_history[0:ix]
            self.current_history = self.current_history[ix:]
            return True
        return False

    def player_stats(self):
        return "($player a=wizard b=human c=male d=charming)"

    def header(self):
        return self.library.read_document('system_header') + '\n' + self.library.read_document('system_actor')

    def mid_pre(self):
        return self.library.read_document('system_mid_pre', player_stats=self.player_stats())

    def mid_post(self):
        return self.library.read_document('system_mid_post', player_stats=self.player_stats())

    def system(self, **kwargs) -> List[str]:
        header : List[str] = self.header().split('\n')
        mid_pre : List[str] = self.mid_pre().split('\n')
        mid_post : List[str] = self.mid_post().split('\n')
        assets = self.shadow.get_assets()
        sheets = list(assets['sheets'])
        dialog = list(assets['dialog'])
        system = header + sheets + mid_pre + dialog + sheets + mid_post
        return self.guidance.format_system(system)

    def __call__(self, processing : Optional[list[str]] = None, **kwargs) -> List[str]:
        self.turn_count = 0

        idp = True
        if processing is None:
            processing = self.current_history
            idp = True

        expansion = self.shadow.expand(processing, **kwargs)
        expansion = [e for e in expansion]
        new_items = [e for e in expansion if e not in processing]
        
        if idp:
            self.recent = {e: self.turn_count for e in new_items}
        
        return self.guidance.format_turn(expansion)

    def expire_recent(self, turn_expiration : int = 4, **kwargs) -> None:
        expire_threshold = self.turn_count - turn_expiration
        self.recent = {k: v for k, v in self.recent.items() if v > expire_threshold}

    def turn(self, turn : list[str], last : int = 1, turn_expriation : int = 4, **kwargs) -> List[str]:
        self.turn_count += 1
        self.expire_recent(turn_expriation, **kwargs)

        expansion = self.shadow.expand(turn, low=False, high=False, **kwargs)
        new_items = [e for e in expansion if e not in turn and e not in self.recent]

        self.recent = {**{e: self.turn_count for e in new_items}, **self.recent}

        return self.guidance.format_turn(turn=new_items + turn[-last:])

    def extract_dialog(self, line : str) -> bool:
        # Inspect our response, to get our char history
        parts = line.split(':', 1)
        if len(parts) > 1:
            char = parts[0].strip()
            self.char_dialog[char].append(parts[1])
            return True
        return False

    def pop(self) -> str:
        while len(self.current_history[-1]) == 0 or self.current_history[-1][0] != '>':
            self.current_history.pop()
        player_input = self.current_history.pop()
        return player_input

    def add_history(self, source : str, turn : str, **kwargs) -> bool:
        self.current_history.append(turn)
        rolled = self.roll_history(**kwargs)
        if rolled == False:
            logger.debug(f"Adding History, {len(self.current_history)}")

    def history_halve(self, current_clearance : int = 0, **kwargs) -> bool:
        return self.roll_history(force_roll=True, **kwargs)

    def get_history(self, expand_history : bool = True, start : int = 0, end : int = -1, **kwargs) -> List[str]:
        history = [*self.past_history, *self.current_history]
        # slice n pages, select stride #
        pages = len(history)
        if end == -1 or end > pages:
            end = pages
        if start > end:
            start = end
        print(f"History: {start} {end}, {pages}")
        band = history[start:end]
        expansion = band if not expand_history else \
            self.shadow.expand(band, low=True, high=True, **kwargs)
        items = [e for e in expansion]
        return items

    @classmethod
    def from_config(cls, **kwargs) -> 'Charmer':
        config = { 
            'resources_path': 'resources',
            'scene_name': 'verana',
            'player': '$player',
            'party': [],
            'location_name': 'Verana',
            'model_guidance': 'dialog',
            **kwargs }
        library = Librarian.from_config(**config)
        shadow = ContextShadowing.from_file(**config)
        guidance = GuidanceStrategy.from_config(**config)
        return cls(library=library, shadow=shadow, guidance=guidance)
    
    def save_game(self, **kwargs) -> str:
        history = self.past_history + self.current_history
        return self.save_game_text(history, **kwargs)

    @classmethod
    def save_game_text(cls, history: list[str], **kwargs) -> str:
        config = {
            'save_file': 'local/savegame.txt',
            **kwargs
        }

        save_file = config['save_file']
        with open(save_file, 'w') as f:
            for h in history:
                f.write(h + '\n')

        return save_file        

    @classmethod
    def load_game_text(cls, **kwargs) -> Optional[list[str]]:
        config = {
            'save_file': 'local/savegame.txt',
            **kwargs
        }

        if not os.path.exists(config['save_file']):
            return None

        def generate_lines():
            with open(config['save_file'], 'r') as f:
                while True:
                    line = f.readline()
                    if line == '':
                        break
                    yield line.strip()
        
        return [line for line in generate_lines()]

    @classmethod
    def save_game_binary(cls, history: list[str], **kwargs):
        config = {
            'save_file': 'local/savegame.dat',
            **kwargs
        }

        with open(config['save_file'], 'wb') as f:
            separator = b'\x1E'  # custom separator byte
            for entry in history:
                f.write(entry.encode() + separator)

    @classmethod
    def load_game_binary(cls, **kwargs) -> Optional[list[str]]:
        config = {
            'save_file': 'local/savegame.dat',
            **kwargs
        }

        if not os.path.exists(config['save_file']):
            return None

        with open(config['save_file'], 'rb') as f:
            separator = b'\x1E'  # custom separator byte
            history = []
            entry = b''
            while True:
                byte = f.read(1)
                if byte == separator:
                    history.append(entry.decode())
                    entry = b''
                elif byte == b'':
                    break
                else:
                    entry += byte

        return history

    def format_history(self, n_show : int = 10, **kwargs) -> str:
        if n_show == -1:
            n_show = len(self.current_history)
        return self.guidance.format_turn(self.current_history[-n_show:], hide_system=True)

    def last_turn(self, include_player : bool = False, **kwargs) -> List[str]:
        return self.get_turn(self.current_history, include_player=include_player, **kwargs)

    @classmethod
    def get_turn(cls, history : List[str], include_player : bool, **kwargs) -> List[str]:
        turn = []
        player_count = 2 if include_player else 1
        for i in range(1,5):
            if i > len(history) or len(history[-i]) == 0:
                continue
            if i > 1 and history[-i][0] == '>':
                player_count -= 1

            if player_count == 0:
                break

            turn.append(history[-i])
        turn = turn[::-1]
        return turn
