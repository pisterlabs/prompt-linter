import textworld # type: ignore
from npc.chain import NPC
from npc.prompts import SHEM
from npc.utils import format_scene, format_command, format_notes, format_toks
from langchain.callbacks import get_openai_callback # type: ignore
import time

STUCK_LENGTH = 2
MEM_LENGTH = 10
TEMP = 0.0
TOKS = 53
class Game():
    """High-level game class that handles the game loop and agent interaction.
    
    Args:
        game_file (str): Path to the game file.
        max_steps (int, optional): Maximum number of steps the game can take. Defaults to 10.
        shem (str, optional): The agent's shem. Defaults to "".
    """
    def __init__(self, game_file, max_steps=10, shem=SHEM):
        infos = textworld.EnvInfos(
            feedback=True,    # Response from the game after typing a text command.
            description=True, # Text describing the room the player is currently in.
            inventory=True,    # Text describing the player's inventory.
            max_score=True,   # Maximum score obtainable in the game.
            score=True,       # Score obtained so far.
        )
        self.world  = textworld.start(game_file, infos=infos)
        self.world.seed = 42
        self.agent = NPC(shem=shem)
        self.shem = shem
        self.max_steps = max_steps
        self.steps = 0
        self.notes = "No notes yet"
        self.log = []
        self.npcs_used = 1
        self.stuck = 0
        self.stuck_threshold = STUCK_LENGTH
        self.stuck_increment = STUCK_LENGTH
        self.shem = self.agent.shem
    
    def get_state(self):
        return dict([item for item in self.world.state.items() if item[0] != 'location'])

    def new_npc(self, shem=SHEM, stuck_length=STUCK_LENGTH, mem_length=MEM_LENGTH, temp=TEMP, toks=TOKS):
        """Create a new NPC agent."""
        memories = {**self.agent.s_chain.memory.dict()['memories'][0]['store']}
        self.agent = None
        self.agent = NPC(shem=shem, memories=memories, mem_length=mem_length, temp=temp, toks=toks)
        self.stuck_threshold = stuck_length
        self.stuck_increment = stuck_length
        self.npcs_used += 1
        notif = "New NPC agent created, #" + str(self.npcs_used)
        print(notif)
        self.log.append(notif)
        self.shem = shem

    def check_stuck(self, command):
        if self.world.state.last_command == command:
            self.stuck += 1
        if self.steps > self.world.state.moves + self.stuck_threshold:
            self.stuck += 1
        if self.stuck > 2:
            self.log.append("Stuck, creating new NPC")
            self.new_npc()
            command = "Look"
            self.stuck_threshold += self.stuck_increment
            self.stuck = 0
        return command
                    
    def step_world(self, command):
        """Send the agent's command to the game world and receive feedback."""
        game_state, _, _ = self.world.step(command)
        self.log.append(format_scene(game_state))
        self.steps += 1
        return game_state
    
    def step_agent(self):
        game_state = self.world.state
        """Send the game state to the agent and receive the agent's command."""
        scene = format_scene(game_state)
        response = self.agent.act(human_input=scene)
        notes = format_notes(response)
        self.notes = notes
        self.log.append(self.notes)
        command = format_command(response)
        self.log.append(command)
        command = self.check_stuck(command)
        response = {'command': command, 'notes': notes}
        return response

    def run(self):
        start = time.time()
        """Step through the game loop, sending the agent's intentions to the game world and receiving feedback."""
        game_state = self.world.reset()
        done = False
        i = 0
        # Start the game loop
        with get_openai_callback() as cb:
            while not done:
                i += 1
                print("#"*50, i)
                print(format_scene(game_state))
                # This is where the action happens
                resp = self.step_agent()
                command = resp['command']
                print(f"({resp['notes']})")
                print(">",command)
                # Step the game world
                game_state = self.step_world(command)
                if i >= self.max_steps:
                    break
            # Game over
            self.world.render()  # Final message.
            print(f"Played {game_state.moves} steps with {self.npcs_used} NPCs, scoring {game_state.score} points.")
            format_toks(cb.total_tokens)
        # End the game loop
        end = time.time()    
        minutes = (end - start) / 60
        print(f"Time elapsed: {minutes:.2f} minutes")

