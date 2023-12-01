import json
import openai
import asyncio

import twilio.rest as twilio

import functions.assistant_functions as assistant_functions
from functions.kasa_controller import KasaController
import assistant.text_classifier as text_classifier

from config.config_variables import api_credentials, name

openai.api_key = api_credentials["openai"]["key"]

class Brain:
    def __init__(self):
        self.initial_prompt_dialog = open("./config/assistant_prompt.txt", "r").readlines()

        self.initial_prompt = f"You are a household assistant AI named Lumo. When you receive a prompt, respond to it in a helpful way. In addition you are able to call simple commands so you have extra functionality. To run one of these commands, it is extremely important you include \"> command_name_here\" in your response. You do not know the current date. The user's name is {name}."

        self.available_commands = json.load(open("./config/assistant_functions.json"))

        self.saved_chats = [
            {"role": "system", "content": self.initial_prompt},
            {"role": "system", "content": "Here are your available commands:" + json.dumps(self.available_commands) },
            {"role": "system", "content": "\n".join(self.initial_prompt_dialog)}
        ]

        self.long_term_memory = []

        self.last_system_chat = len(self.saved_chats)

        self.kasa_controller = KasaController()
        self.twilio_client = twilio.Client(api_credentials["twilio"]["sid"], api_credentials["twilio"]["auth_token"])

    def update_data(self, data):
        data = json.loads(data)
        for row in data:
            self.saved_chats.append(row)

    def clear_chat(self):
        while len(self.saved_chats) > (self.last_system_chat):
            self.saved_chats.pop(self.last_system_chat)

    def makeRequest(self, messageBody, room_name, role="user", server=None ):
        new_chats = []

        new_chats.append( {"role": role, "content": messageBody} )

        if text_classifier.is_important(messageBody):
            print("Important: ", messageBody)
            self.long_term_memory.append({"role": role, "content": messageBody})

            while len(self.long_term_memory) > 15:
                self.long_term_memory.pop(0)

        while len(self.saved_chats+new_chats) > (self.last_system_chat + 15):
            self.saved_chats.pop(self.last_system_chat)

        chat_completion = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=self.saved_chats+new_chats)

        self.saved_chats.append(chat_completion.choices[0].message)

        lines = chat_completion.choices[0].message.content.splitlines()

        parsed_lines = []

        for line in lines:            
            if not line or line.isspace():
                continue
            if not ">" in line:
                result = {"role": "assistant", "content": line}
                parsed_lines.append( result )
                new_chats.append(result)
                continue

            if not "> " in line:
                line.replace(">", "> ")
            
            # Else line is a command
            command = line.split(" ")
            
            #format command so command[2] is arg1, command[3] is arg2, etc
            arg = 2
            while len(command) > arg: 
                if '"' in command[arg]:
                    if command[arg].count("\"") == 1:
                        while len(command) > arg+1 and not '"' in command[arg+1]:
                            command[arg] += " " + command[arg+1]
                            del command[arg+1]
            
                        command[arg] += " " + command[arg+1]
                        del command[arg+1]
                    command[arg] = command[arg][1:-1]

                arg += 1
            
            if command[1] == "get_time":
                if len(command) == 2:
                    result = {"role": "system", "content": assistant_functions.get_time()}
                    parsed_lines.append( result )
                    new_chats.append(result)

                if len(command) == 3:
                    result = {"role": "system", "content": assistant_functions.get_time_at(command[2])}
                    parsed_lines.append( result )
                    new_chats.append(result)
            
            elif command[1] == "get_weather":                
                if len(command) == 3:
                    result = {"role": "system", "content": assistant_functions.get_weather(command[2])}
                    parsed_lines.append( result )
                    new_chats.append(result)

                if len(command) == 4:
                    result = {"role": "system", "content": assistant_functions.get_weather_at(command[2], command[3])}
                    parsed_lines.append( result )
                    new_chats.append(result)
            
            elif command[1] == "send_text":
                if len(command) == 4:
                    result = {"role": "system", "content": assistant_functions.send_text(self.twilio_client, command[2], command[3])}
                    parsed_lines.append( result )
                    new_chats.append(result)
            
            elif command[1] == "search_internet":
                if len(command) == 3:
                    result = {"role": "system", "content": assistant_functions.search_web(command[2])}
                    parsed_lines.append( result )
                    new_chats.append(result)

            elif command[1] == "generate_image":
                print(command)
                if len(command) == 3:
                    img = assistant_functions.generate_image(command[2])
                    new_chats.append({"role": "system", "content": img[0]})
                    if len(img) > 1:
                        parsed_lines.append({"role": "image", "content": img[1]})
            
            elif command[1] == "find_nearby_locations":
                if len(command) == 3:
                    result = {"role": "system", "content": assistant_functions.find_nearby_locations(command[2])}
                    parsed_lines.append( result )
                    new_chats.append(result)

                if len(command) == 4:
                    result = {"role": "system", "content": assistant_functions.find_nearby_locations(command[2], command[3])}
                    parsed_lines.append( result )
                    new_chats.append(result)
            
            elif command[1] == "smart_device_toggle":
                if len(command) == 4:
                    asyncio.run(self.kasa_controller.set_plug(name=command[3], on=command[2]))
                    new_chats.append({"role": "system", "content": f"lights in room {command[3]} switched {command[2]}"})

            elif command[1] == "room_light_toggle":
                print(command)
                if len(command) == 3:
                    asyncio.run(self.kasa_controller.set_room(name=room_name, on=command[2]))
                    new_chats.append({"role": "system", "content": f"lights in room {room_name} switched {command[2]}"})

                if len(command) == 4:
                    asyncio.run(self.kasa_controller.set_room(name=command[3], on=command[2]))
                    new_chats.append({"role": "system", "content": f"lights in room {command[3]} switched {command[2]}"})

            elif command[1] == "room_light_brightness":
                print(command)
                if len(command) == 3:
                    asyncio.run(self.kasa_controller.set_room(name=room_name, brightness=command[2]))
                    new_chats.append({"role": "system", "content": f"lights in room {room_name} set to brightness {command[2]}"})

                if len(command) == 4:
                    asyncio.run(self.kasa_controller.set_room(name=command[3], brightness=command[2]))
                    new_chats.append({"role": "system", "content": f"lights in room {command[3]} set to brightness {command[2]}"})

            elif command[1] == "room_light_brightness_adjust":
                print(command)
                if len(command) == 3:
                    asyncio.run(self.kasa_controller.adjust_room_brightness(name=room_name, dir=command[2]))
                    new_chats.append({"role": "system", "content": f"lights in room {room_name} adjusted {command[2]} by default (20)"})

                if len(command) == 4:
                    if int(command[3]):
                        asyncio.run(self.kasa_controller.adjust_room_brightness(name=room_name, dir=command[2], brightness=command[3]))
                        new_chats.append({"role": "system", "content": f"lights in room {room_name} adjusted {command[2]} by {command[3]}"})
                    else:
                        asyncio.run(self.kasa_controller.adjust_room_brightness(name=command[3], dir=command[2]))
                        new_chats.append({"role": "system", "content": f"lights in room {command[3]} adjusted {command[2]} by default of 20"})

                if len(command) == 5:
                    asyncio.run(self.kasa_controller.adjust_room_brightness(name=command[4], dir=command[2], brightness=command[3]))
                    new_chats.append({"role": "system", "content": f"lights in room {command[4]} adjusted {command[2]} by {command[3]}"})

            elif command[1] == "room_light_color":
                print(command)
                if len(command) == 3:
                    asyncio.run(self.kasa_controller.set_room(name=room_name, color=command[2]))
                    new_chats.append({"role": "system", "content": f"lights in room {room_name} set to color {command[2]}"})

                if len(command) == 4:
                    asyncio.run(self.kasa_controller.set_room(name=command[3], color=command[2]))
                    new_chats.append({"role": "system", "content": f"lights in room {command[3]} set to color {command[2]}"})

            elif command[1] == "control_music":
                new_chats.append({"role": "system", "content": f"Setting music to {' '.join(command[2:])}"})
                parsed_lines.append({"role": "music", "content": " ".join(command[2:]) })

            elif command[1] == "set_alarm_static": 
                print(command)
                if len(command) == 3:
                    # one-time alarm
                    result = {"role": "system", "content": assistant_functions.set_alarm_static(command[2])}
                    new_chats.append(result)

                if len(command) == 4:
                    # repeating alarm
                    result = {"role": "system", "content": assistant_functions.set_alarm_static_at(command[2], command[3])}
                    new_chats.append(result)

            elif command[1] == "remove_alarm_static":
                print(command)
                if len(command) == 3:
                    # one-time alarm
                    result = {"role": "system", "content": assistant_functions.remove_alarm_static(command[2])}
                    new_chats.append(result)

                if len(command) == 4:
                    # repeating alarm
                    result = {"role": "system", "content": assistant_functions.remove_alarm_static_at(command[2], command[3])}
                    new_chats.append(result)

            else:
                print(command)
                result = {"role": "assistant", "content": line}
                parsed_lines.append( result )
                new_chats.append(result)

        server.update_all_servers(new_chats)

        self.saved_chats += new_chats
        while len(self.saved_chats) > (self.last_system_chat + 15):
            self.saved_chats.pop(self.last_system_chat)
        
        return parsed_lines