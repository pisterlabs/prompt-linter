
class ConversationManager:

    def __init__(self):
        pass

    def respond_to_player(self, player_input):
        return player_input
    

class GPTConversationManager(ConversationManager):
    import openai
    import json

    def __init__(self):
        self.openai_info = self._load_openai_info()
        super().__init__()


    def generate_response_with_character_template(
            self, player_input: str, character_name: str, character_description: str,
            player_name: str, player_description: str,
            history_summary: str, conversation_tuple_list: list
    ) -> str:
        
        prompt_template = open('./assets/prompts/characters/character_template.txt').read()
        prompt_template = prompt_template.replace("__CHARACTER_NAME__", character_name)
        prompt_template = prompt_template.replace("__CHARACTER_DESCRIPTION__", character_description)
        prompt_template = prompt_template.replace("__PLAYER_NAME__", player_name)
        prompt_template = prompt_template.replace("__PLAYER_DESCRIPTION__", player_description)
        prompt_template = prompt_template.replace("__HISTORY_SUMMARY__", history_summary)
        conversation_string = self.generate_conversation_string_from_tuple_list(conversation_tuple_list)
        prompt_template = prompt_template.replace("__CONVERSATION__", conversation_string)
        prompt_template = prompt_template.replace("__PLAYER_INPUT__", player_input)

        # print(prompt_template)

        messages = [{
            'role': 'user',
            'content': prompt_template
        }]

        completion = self.openai.Completion.create(
            model="davinci", 
            prompt=prompt_template,
            stop=[f"{player_name}:"],
            max_tokens=256,
            )
            
        return completion.choices[0]['text']
    
    
    def add_input_and_response_to_conversation_tuple_list(
            self, player_input: str, character_name: str, character_description: str,
            player_name: str, player_description: str,
            history_summary: str, conversation_tuple_list: list
    ) -> list:
        response = self.generate_response_with_character_template(
            player_input, character_name, character_description,
            player_name, player_description,
            history_summary, conversation_tuple_list          
        )
        conversation_tuple_list.append((player_name, player_input))
        conversation_tuple_list.append((character_name, response.replace("\n", "").strip()))
        return conversation_tuple_list
        

    def generate_conversation_string_from_tuple_list(
            self,
            conversation_tuple_list: list
    ) -> str:
        conversation_string = ""
        for message in conversation_tuple_list:
            conversation_string += (
                message[0].replace("\n","") + ": " + message[1].replace("\n", "") + "\n"
                )
        return conversation_string


    def generate_response_with_messages(
            self,
            player_input: str,
            npc_conversation_history: list,
            npc_description: str,
            npc_history_summary: str,
            player_name: str = "player",
            debug: bool = False
            ):
        
        
        game_instructions = {
            "role": "user", "content": "You are a non-player character in a text adventure game. Interact with the user as if they are playing the game. You shall not make reference to the game. You shall not act as an assistant. You shall act in accordance with the character traits that will be described to you. It is extremely important that the user not be made aware that this is a game."
            }
        
        messages = [
            game_instructions,
            {"role": "user", "content": npc_description}, 
            {"role": "user", "content": npc_history_summary}
            ]
        
        messages += npc_conversation_history
        messages += [{"role": "user", "content": player_input}]

        if debug:
            print(messages)

        completion = self.openai.ChatCompletion.create(
            model="gpt-3.5-turbo", 
            messages=messages
            )
        
        npc_conversation_history += [{"role": "user", "content": player_input}]
        npc_conversation_history += [{"role": 'system', "content": completion.choices[0].message.content}]
        print(completion.choices[0].message.content)    
        return npc_conversation_history


    def _load_openai_info(self) -> dict:
        f = open(".local/openai.json", "r")
        openai_info = self.json.load(f)
        f.close()
        self.openai.api_key = openai_info['api_key']
        return openai_info
    
if __name__ == "__main__":
    cm = GPTConversationManager()
    npc_description = "You are a physical representation of an operating system. Other programs can interact with you as though you are a sentient being. You physical appearance is humanoid-like. You emit a blue glow."
    npc_name = "Cyber"
    history = []
    history_summary = "This is your first time meeting the user. The user is a player in a text adventure game. You are to interact with the user as a character in the game."
    player_name = "Tron"
    player_description = "Tron is a human embodiment of a computer program."
    while True:
        user_input = input(">")
        history = cm.add_input_and_response_to_conversation_tuple_list(
            user_input, npc_name, npc_description,
            player_name, player_description,
            history_summary, history
        )
        print(history[-1][1])
