from langchain.callbacks.base import BaseCallbackHandler


class UpdateDiscordMessageHandler(BaseCallbackHandler):
    name = "update_discord_message_handler"

    def handle_llm_new_token(self, token: str):
        # Send the token to your desired destination here (e.g., Discord bot)
        print("token", token)
