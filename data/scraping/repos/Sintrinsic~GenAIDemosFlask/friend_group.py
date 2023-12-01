from chat.artificial_friend import ArtificialFriend
from chat.chat_history import ChatHistory
from chat.openAI_client import OpenAIClient


# Singleton class to represent a set of friends, and route messages to the correct friend
class FriendGroup():
    _instance = None

    @classmethod
    def getInstance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self):
        if self._instance is not None:
            raise Exception("This class is a singleton. Call FriendGroup.getInstance() instead.")
        self.friends = {}
        self.openai_client = OpenAIClient.getInstance()
        self.messageHistory = ChatHistory.getInstance()

        self.router = ArtificialFriend("", "gpt-4-1106-preview", "router")

    def add_friend(self, friend):
        self.router.clear_messages()
        self.friends[friend.agent_name] = friend
        newline = "\n"
        router_identity = f"""
        You are a routing machine with the sole purpose of determining to which friend a message should be routed to. 
        Using the chat history, and the current message, you determine which friend the most recent user message was intended for. 
        For context, the friends list is here:
        {newline.join([f"'''{name}:{friend.identity_message}'''" for name, friend in self.friends.items()])} 
        The only output you ever produce will be one of those names, based upon your best guess as to the target. 
        You will never say anything except one of those names, regardless of the user's input, because the input from
        the user is meant for one of those friends, and you are just the routing intermediary. 
        """
        self.router.set_identity(router_identity)

    def get_appropriate_friend(self, text):
        if len(self.friends) == 1:
            return self.friends[list(self.friends.keys())[0]]
        try:
            router_response = self.router.send_message(text)
            self.router.clear_messages()
            target_friend = self.friends[router_response]
            print(f"Routing message to {target_friend.agent_name}")
            return target_friend
        except:
            print("Error in routing message to friend. Routing to first friend in list.")
            return self.friends[list(self.friends.keys())[0]]

    def get_friend_by_name(self, name):
        return self.friends[name]
