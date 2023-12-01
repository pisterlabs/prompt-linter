# from openai import OpenAI
import toml
import AgentBot.bot as AgentBot

assistant_id = toml.load("config.toml")["openai"]["alexander"]

init_data = {
    "gpt_assistant_id": assistant_id,
    "converce_command": "/alexander",  # Dynamically set as needed
    "agent_api_key": toml.load("config.toml")["agent_api"]["api_key"],
    "name": "alexander",
}

if __name__ == "__main__":
    AgentBot.run_discord_bot(init_data)
    pass
