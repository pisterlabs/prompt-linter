# valai/pinnacle/charmer.py

from collections import defaultdict
import logging
import os
from typing import List, Optional, Dict

from .director import SceneDirector, sample_response
from .exception import DirectorError
from .guidance import GuidanceStrategy
from .prompt import Librarian
from .scene import DirectorDialog
from .symbol import ContextShadowing

logger = logging.getLogger(__name__)


class DirectorCharmer:
    """The Charmer is the main interface to the Charm system. It is responsible for
    managing the game state, and the interaction between the player and the system.
    """
    def __init__(self, director : SceneDirector, library : Librarian, shadow : ContextShadowing,
                  guidance : GuidanceStrategy, history_threshold : int = 100) -> None:
        self.director = director
        self.library = library
        self.shadow = shadow
        self.turn_count = 0
        self.recent = {}
        self.history_threshold = history_threshold
        self.past_history = []
        self.current_history = []
        self.char_dialog = defaultdict(list)
        self.guidance = guidance
        self.character_dialog = director.sym.character_dialog

    def init_history(self, load : bool = False, **kwargs) -> bool:
        history = None
        if load: 
            history = self.load_game_text()
        initial_q = "$player (to ZxdrOS, restart): New Game"
        initial_a = "ZxdrOS (to $player, announcing): *Nodding*  The world is made new again.  Welcome to Novara."
        initial_l = [ loc.travel_line(self.director.roster.player.sheet) for k, loc in self.director.sym.locations.items() if loc.start ]
        if history is None:
            history = [initial_q, initial_a] + initial_l
        self.past_history = []
        self.current_history = history

        # Process the initial history
        self.lite_reset(**kwargs)

        return self.roll_history(**kwargs)

    def lite_reset(self, **kwargs) -> bool:
        # We need to extract any major state changes from the history

        for line in self.past_history:
            action = sample_response(line=line, **kwargs)
            self.handle_action(action, skip_location=True, **kwargs)
        for line in self.current_history:
            action = sample_response(line=line, **kwargs)
            self.handle_action(action, skip_location=True, **kwargs)
        logger.debug(f"Processed {len(self.past_history)} past history, {len(self.current_history)} current history")

        return True

    def reset(self, **kwargs) -> bool:
        # Resetting needs to clear all of the director / scene state, and then re-process all of the system
        # and scene history to get back to the current state.
        # First we need to extract any major state changes from the history
        # Then we need to reset the director
        # Then we need to re-process the system lines
        # Then we need to re-process the scene lines
        # Then we need to re-process the past history
        # Then we need to re-process the current history
        # Then we need to roll the history
        scene = self.discover_scene(**kwargs)
        if scene is None:
            logger.info("No scene found in history, resetting to start")
            initial_l = [ loc for k, loc in self.director.sym.locations.items() if loc.start ]
            if len(initial_l) == 0:
                raise DirectorError("No start location found")
            location = initial_l[0]
            scene = location.symbol

        self.director.reset_scene()
        self.set_scene(scene, quiet=True, **kwargs)
        lines = self.system_lines(**kwargs)
        lines += self.scene_lines(**kwargs)
        for line in lines:
            action = sample_response(line=line, **kwargs)
            self.handle_action(action, **kwargs)
        for line in self.past_history:
            action = sample_response(line=line, **kwargs)
            self.handle_action(action, **kwargs)
        for line in self.current_history:
            action = sample_response(line=line, **kwargs)
            self.handle_action(action, **kwargs)
        logger.debug(f"Processed {len(lines)} scene lines, {len(self.past_history)} past history, {len(self.current_history)} current history")

        return True

    def roll_history(self, force_roll : bool = False, **kwargs) -> bool:
        # we want to preserve # Codex
        if len(self.current_history) > self.history_threshold or force_roll:
            # cut current history in half
            ix = len(self.current_history)//2
            codex = [e for e in self.current_history if e[:5] == 'Codex']
            self.past_history += self.current_history[0:ix]
            self.current_history = self.current_history[ix:]
            return True
        return False

    def header(self):
        return self.library.read_document('system_header') + '\n' + self.library.read_document('system_actor')

    def mid_pre(self):
        return self.library.read_document('system_mid_pre', player_stats=self.director.roster.player.sheet.stat_line())

    def mid_post(self):
        return self.library.read_document('system_mid_post', player_stats=self.director.roster.player.sheet.stat_line())

    def system_lines(self, **kwargs) -> List[str]:
        if self.director.scene is None:
            raise Exception("No scene loaded")
        header : List[str] = self.header().split('\n')
        mid_pre : List[str] = self.mid_pre().split('\n')
        mid_post : List[str] = self.mid_post().split('\n')

        all_stats = []

        for c, character in self.director.roster.characters.items():
            all_stats.append(character.sheet.character_line())
            all_stats.append(character.sheet.description_line())

        system = header + mid_pre + mid_post
        return system

    def system_header(self, **kwargs) -> str:
        lines = self.system_lines(**kwargs)
        return self.guidance.format_system(lines, output_end=False, **kwargs)

    def scene_lines(self, **kwargs) -> List[str]:
        if self.director.scene is None:
            raise Exception("No scene loaded")
        
        scene_stats, party_stats, dialog = [], [], []

        for actor in self.director.roster.get_party().values():
            party_stats.append(actor.sheet.character_line())
            party_stats.append(actor.sheet.description_line())
            party_stats.append(f"[{actor.sheet.symbol} - {actor.sheet.name} is following $player]")
            party_stats += actor.sheet.sales_lines()
            party_stats += actor.sheet.equipment_lines()
            party_stats += actor.sheet.spell_lines()
            dialog += self.character_dialog(actor.sheet)

        for actor in self.director.roster.get_actors(self.director.scene.characters).values():
            scene_stats.append(actor.sheet.character_line())
            scene_stats.append(actor.sheet.description_line())
            scene_stats += actor.sheet.sales_lines()
            scene_stats += actor.sheet.equipment_lines()
            scene_stats += actor.sheet.spell_lines()
            dialog += self.character_dialog(actor.sheet)

        player = self.director.scene.get_player_prompt()
        scene = self.director.scene.get_location_prompt()
        system = dialog + scene_stats + player + party_stats + scene
        return system
    
    def scene_header(self, **kwargs) -> str:
        lines = self.scene_lines(**kwargs)
        return self.guidance.format_system(lines, output_head=False, **kwargs)

    def set_scene(self, location_symbol : str, quiet : bool = True, **kwargs) -> None:
        logger.debug(f"Setting Scene to {location_symbol}")
        self.director.set_scene(location_symbol, **kwargs)
        self.char_dialog = defaultdict(list)
        self.turn_count = 0
        self.shadow.reload(location_symbol, party=self.director.roster.party, **kwargs)
        if quiet == False:
            self.add_history('system', self.director.scene.location.travel_line(self.director.roster.player.sheet), **kwargs)

    def discover_scene(self, **kwargs) -> Optional[str]:
        location_symbol = None
        for i in range(len(self.current_history)):
            entry = self.current_history[-i]
            result = sample_response(line=entry, **kwargs)
            match_type = result.get('match', '')
            actor = result.get('actor', '')
            if match_type == 'location' and actor == self.director.roster.player.sheet.symbol:
                logger.debug(f"Discover Scene: {result}")
                location_symbol = result.get('location', None)
            if location_symbol is not None:
                break
        if location_symbol is None:
            for i in range(len(self.past_history)):
                entry = self.past_history[-i]
                result = sample_response(line=entry, **kwargs)
                match_type = result.get('match', '')
                actor = result.get('actor', '')
                if match_type == 'location' and actor == self.director.roster.player.sheet.symbol:
                    location_symbol = result.get('location', None)
                if location_symbol is not None:
                    break
        return location_symbol

    def __call__(self, processing : Optional[List[str]] = None, idp : bool = False, tick : bool = True, **kwargs) -> List[str]:
        if tick: 
            # TODO I guess we should count the turns here
            self.turn_count = 1

        if processing is None:
            processing = self.current_history
            idp = True

        expansion = self.shadow.expand(processing, **kwargs)
        expansion = [e for e in expansion]
        #logger.debug(f"Expanding {processing} to {expansion}")
        
        if idp:
            self.recent = {e: self.turn_count for e in expansion if e not in processing}
        
        return self.guidance.format_turn(turn=expansion)

    def uncharmed(self, hold : int = 3, take : int = 1, **kwargs) -> List[str]:
        ptr = len(self.current_history) - 1
        if hold > 0:
            hix = hold
        elif take > 0:
            hix = take

        while ptr > 0 and hix > 0:
            if self.current_history[ptr][0] == '$':
                hix -= 1
            ptr -= 1

        if hold > 0:
            history = self.current_history[0:ptr]
        else:
            history = self.current_history[ptr:]

        return self.guidance.format_turn(turn=history)

    def replay(self, hold : int = -1, take : int = 3, **kwargs) -> List[str]:
        ptr = len(self.current_history) - 1
        if hold > 0:
            hix = hold
        elif take > 0:
            hix = take

        while ptr > 0 and hix > 0:
            if self.current_history[ptr][0] == '$':
                hix -= 1
            ptr -= 1

        if hold > 0:
            history = self.current_history[0:ptr]
        else:
            history = self.current_history[ptr:]
            
        return self(processing=history, tick = False, idp=True, **kwargs)

    def expire_recent(self, turn_expiration : int = 4, **kwargs) -> None:
        expire_threshold = self.turn_count - turn_expiration
        self.recent = {k: v for k, v in self.recent.items() if v > expire_threshold}

    def turn(self, turn : list[str], last : int = 1, turn_expriation : int = 4, **kwargs) -> str:
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

    def handle_action(self, action : Dict[str, str], skip_location : bool = False, **kwargs) -> bool:
        # TODO eventually replace this with a clever kwargs based dispatch

        match_type = action.get('match', '')

        if match_type == '' or match_type in ['dialog', 'empty', 'system', 'symbol', 'statement']:
            return False

        elif match_type == 'location':
            location = action.get('location', '')
            #logger.debug(f"Handling Action: {action}")
            if skip_location:
                return False

            self.set_scene(location_symbol=location, **kwargs)
            return True

        elif match_type == 'item':
            actor = action.get('actor', '')
            gl = action.get('gained_lost', '')
            if actor == self.director.roster.player.sheet.symbol:
                if gl == 'has gained':
                    #logger.debug(f"Handling Action: {action}")
                    self.director.roster.player.add_item(**action, **kwargs)
                    return True
                elif gl == 'has lost':
                    #logger.debug(f"Handling Action: {action}")
                    self.director.roster.player.remove_item(**action, **kwargs)
                    return True
            scene = self.director.scene
            if scene is not None and actor in scene.characters:
                if gl == 'has gained':
                    #logger.debug(f"Handling Action: {action}")
                    scene.roster.get_actor(actor).add_item(**action, **kwargs)
                    return True
                elif gl == 'has lost':
                    #logger.debug(f"Handling Action: {action}")
                    scene.roster.get_actor(actor).remove_item(**action, **kwargs)
                    return True
            elif scene is not None:
                logger.debug(f"Item Action: {action} not in characters.")

        elif match_type == 'party':
            #logger.debug(f"Handling Action: {action}")
            name = action.get('name', '')
            actor = action.get('actor', '')
            join = action.get('join', '')
            if actor != '':
                if join == 'has joined':
                    logger.debug(f"Party Join: {name} joins {self.director.roster.player_name}'s party.")
                    self.director.roster.add_to_party(actor)
                    return True
                elif join == 'has left':
                    logger.debug(f"Party Leave: {name} leaves {self.director.roster.player_name}'s party.")
                    self.director.roster.remove_from_party(actor)
                    return True
            else:
                logger.debug(f"Party Action: {action} not in characters.")
                return True
        
        logger.debug(f"Unhandled Action: {action}")
        return False

    def add_residue(self, residue : List[str], **kwargs) -> bool:
        for line in residue:
            action = sample_response(line=line, **kwargs)
            self.handle_action(action, **kwargs)
            self.add_history('system', line, **kwargs)
        return self.roll_history(**kwargs)

    def add_history(self, source : str, turn : str, **kwargs) -> bool:
        action = sample_response(line=turn, **kwargs)
        self.handle_action(action, **kwargs)
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
        logger.debug(f"History: {start} {end}, {pages}")
        band = history[start:end]
        expansion = band if not expand_history else \
            self.shadow.expand(band, low=True, high=True, **kwargs)
        items = [e for e in expansion]
        return items

    @classmethod
    def from_config(cls, **kwargs) -> 'Charmer':
        character_dialog = DirectorDialog()
        director = SceneDirector.from_config(character_dialog=character_dialog, **kwargs)
        library = Librarian.from_config(**kwargs)
        shadow = ContextShadowing.from_config(character_dialog=character_dialog, **kwargs)
        guidance = GuidanceStrategy.from_config(**kwargs)
        return cls(director=director, library=library, shadow=shadow, guidance=guidance)
    
    def save_game(self, **kwargs) -> str:
        history = self.past_history + self.current_history
        return self.save_game_text(history, **kwargs)

    @classmethod
    def save_game_text(cls, history: list[str], **kwargs) -> str:
        config = {
            'save_file': 'local/pinnacle_savegame.txt',
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
            'save_file': 'local/pinnacle_savegame.txt',
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
            hix = len(history) - i
            if hix < 0:
                break
            if i > 1 and history[hix][0] in ('>', '$'):
                player_count -= 1

            if player_count == 0:
                break

            turn.append(history[hix])
        turn = turn[::-1]
        return turn