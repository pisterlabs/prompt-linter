from collections import defaultdict
from openai.error import ServiceUnavailableError
from typing import Optional
import backoff
import openai
import random
import telnetlib

TELNET_IP = "127.0.0.1"
MSG_STRING = " : "
DEAD_MSG = "*DEAD*"

MAX_CHAT_SIZE = 128
MIN_CHAT_SIZE = 2

T_CHAT_PREFIX = "(Terrorist)"
CT_CHAT_PREFIX = "(Counter-Terrorist)"

class CSGOChatBot:
    def __init__(self, telnet_port: int, player_name: str, chat_history_size: int):
        self.telnet_port = telnet_port
        self.player_name = player_name
        self.message_history_per_name = defaultdict(list)
        self.chat_history_size = chat_history_size

    def set_openai_key(self, api_key: str) -> None:
        openai.api_key = api_key

    @backoff.on_exception(
        backoff.constant,
        ServiceUnavailableError,
        max_time=60
    )
    def get_response_for_message(self, sender: str, msg: str) -> Optional[str]:
        rules = [
            "You are a human playing CSGO.",
            "Keep your message short and funny.",
            "Never mention that you are an AI.",
            "Insert smiley faces and emojis, prefer ascii emojis",
            "No hashtags, Never use the character '#'",
            f"The maximum length of the response is {random.randint(MIN_CHAT_SIZE, MAX_CHAT_SIZE)}.",
            "Respond only to the Current message, but you may take context from the previous messages,",
            "Don't add the sender's name in the response, just the message.",
            "Don't use any punctuation.",
            "Don't use ':'",
        ]

        gpt_messages = [
             {"role": "system", "content": rule} for rule in rules
            ]
        
        if self.chat_history_size > 0:
            history = "Previous messages -> \n" 
            history += "\n".join(self.message_history_per_name[sender])
            gpt_messages.append({"role": "user", "content": history})
            
            if len(self.message_history_per_name[sender]) > self.chat_history_size:
                self.message_history_per_name[sender].pop(0)
            
            self.message_history_per_name[sender].append(msg)
        
        current_message = f"Current message -> \n{sender}: {msg}"
        gpt_messages.append({"role": "user", "content": current_message})
        
        completion = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=gpt_messages,
        )

        response = completion.choices[0].message.content
        return response

    def connect_to_server(self) -> telnetlib.Telnet:
        t = telnetlib.Telnet(TELNET_IP, self.telnet_port)
        t.write(b"say ::::: pycsgogpt started! :::::\n")
        return t

    def process_message(self, t: telnetlib.Telnet, output: str) -> None:
        player_info, player_msg = output.split(MSG_STRING, maxsplit=1)
        is_team = T_CHAT_PREFIX in player_info or CT_CHAT_PREFIX in player_info
        player_name = player_info.strip().replace(DEAD_MSG, "").replace(T_CHAT_PREFIX, "").replace(CT_CHAT_PREFIX, "")

        if self.player_name in player_name:
            return
        
        msg = player_msg.strip()
        print(f"- {player_name}: {msg}")

        response = self.get_response_for_message(player_name, msg)
        
        # Replace anything that can be used for command injection in valve's source engine
        denylist = [";", "&", "|", ">", "<", "$", "`", "\\", "\n", "\r", "\t", "\0", "\x0b", "\x0c", "\'", "\""]
        for char in denylist:
            response = response.replace(char, "")
        
        if response:
            say_cmd = "say_team" if is_team else "say"
            t.write(f'{say_cmd} "{response}"\n'.encode("utf-8"))

    def loop(self) -> None:
        t = self.connect_to_server()
        while True:
            try:
                output = t.read_until(b"\n", timeout=0.1).decode("utf-8")
            except UnicodeDecodeError:
                continue
            
            if not output:
                continue

            if MSG_STRING not in output:
                continue

            self.process_message(t, output)
