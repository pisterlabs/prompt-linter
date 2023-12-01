import autogen
import queue
import openai


openai.log='debug'

config_list = [
    {
        "model": "gpt-4",
        "api_key": "<YOUR KEY HERE>"
    }
]
llm_config = {
    "model":"gpt-4",
    "temperature": 0,
    "config_list": config_list,
        "functions": [
        {
            "name": "search_db",
            "description": "Search database for order status",
            "parameters": {
                "type": "object",
                "properties": {
                    "order_number": {
                        "type": "integer",
                        "description": "Order number",
                    },
                    "customer_number": {
                        "type": "string",
                        "description": "Customer number",
                    }
                },
                "required": ["order_number","customer_number"],
            },
        },
    ],
}

class UserProxyWebAgent(autogen.UserProxyAgent):
    def __init__(self, *args, **kwargs):
        super(UserProxyWebAgent, self).__init__(*args, **kwargs)

    def set_queues(self, client_sent_queue, client_receive_queue):
        self.client_sent_queue = client_sent_queue
        self.client_receive_queue = client_receive_queue

    # this is the method we override to interact with the chat
    def get_human_input(self, prompt: str) -> str:
        last_message = self.last_message()
        if last_message["content"]:
            self.client_receive_queue.put(last_message["content"])
            reply = self.client_sent_queue.get(block=True)
            if reply == "exit":
                self.client_receive_queue.put("exit")
            return reply
        else:
            return


#############################################################################################
# this is where you put your Autogen logic, here I have a simple 2 agents with a function call
class AutogenChat():
    def __init__(self, chat_id=None, websocket=None):
        self.websocket = websocket
        self.chat_id = chat_id
        self.client_sent_queue = queue.Queue()
        self.client_receive_queue = queue.Queue()

        self.assistant = autogen.AssistantAgent(
            name="assistant",
            llm_config=llm_config,
            system_message="""You are a helpful assistant, help the user find the status of his order. 
            Only use the tools provided to do the search. Only execute the search after you have all the information needed. 
            Ask the user for the information you need to perform the search."""
        )
        self.user_proxy = UserProxyWebAgent(  ###### use UserProxyWebAgent
            name="user_proxy",
            human_input_mode="ALWAYS", ######## YOU NEED TO KEEP ALWAYS
            max_consecutive_auto_reply=10,
            is_termination_msg=lambda x: x.get("content", "") and x.get("content", "").rstrip().endswith("TERMINATE"),
            code_execution_config=False,
            llm_config=llm_config,
            system_message="""Reply TERMINATE if the task has been solved at full satisfaction.
        Otherwise, reply CONTINUE, or the reason why the task is not solved yet."""
        )

        # add the queues to communicate 
        self.user_proxy.set_queues(self.client_sent_queue, self.client_receive_queue)

        self.user_proxy.register_function(
            function_map={
                "search_db": self.search_db
            }
        )

    def set_thread(self, thread):
        self.thread = thread

    def start(self):
        data = self.client_sent_queue.get(block=True)
        self.user_proxy.initiate_chat(
            self.assistant,
            message=data
        )

    #MOCH Function call
    def search_db(self, order_number=None, customer_number=None):
        return "Order status: delivered. TERMINATE"
