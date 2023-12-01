import os
import time

import threading
import functools

import pymumble_py3 as pymumble
import PyChatGPT


class GPTMumbleBot:

    def __init__(self):
        self.gpt_response = "No Answer from OpenAI!"
        self.openai_conn = PyChatGPT.ChatGPT(api_key=os.environ["OPENAI_API_KEY"])
        self.setup_Mumble_Connection()

    def setup_Mumble_Connection(self):
        host = os.environ["MumbleServer"]
        port = 64738
        user = "gpt"
        password = os.environ["MumbleServerPWD"]
        certfile = os.environ["MumbleCertPath"]
        keyfile = os.environ["MumbleCertKeyPath"]
        self.mumble_conn = pymumble.Mumble(host, user, port, password, certfile, keyfile)
        self.mumble_conn.start()

        self.msg_rcvd_event = threading.Event()
        self.msg_rcvd_event_callback = functools.partial(self.message_received, self.msg_rcvd_event)

        self.mumble_conn.is_ready()
        self.mumble_conn.callbacks.set_callback(pymumble.constants.PYMUMBLE_CLBK_TEXTMESSAGERECEIVED, self.msg_rcvd_event_callback)

    def message_received(self, event, message):
        self.msg_rcvd_event.set()
        if len(message.message) > 7 and message.message[:7] == "Hey GPT":
            user_request = message.message[7:]
            self.sendAI_answer(user_request)
        else:
            self.msg_rcvd_event.clear()

    def sendAI_answer(self, user_request):
        self.gpt_response = self.openai_conn.human_request(user_request)
        self.mumble_conn.channels.find_by_name(os.environ["MumbleChannelName"]).send_text_message(self.gpt_response)
        self.msg_rcvd_event.clear()

    def wait_for_request(self):
        self.msg_rcvd_event.wait()


if __name__ == "__main__":

    MumbleBot = GPTMumbleBot()
    try:
        while MumbleBot.mumble_conn.is_alive() and not MumbleBot.mumble_conn.exit:
            MumbleBot.wait_for_request()
    except KeyboardInterrupt:
        exit("\nexit()")
