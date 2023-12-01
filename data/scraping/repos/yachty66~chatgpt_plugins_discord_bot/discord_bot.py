import discord
from pathlib import Path
import config
import json
import openai 
import importlib.util

intents = discord.Intents.all()
bot = discord.Bot(intents=intents)
openai.api_key = config.openai_key

class DiscordBot():
    def __init__(self):
        self.register_commands()
        self.plugins = []
        self.run()
        
    def register_commands(self):
        @bot.slash_command(name="chat", description="Use chatgpt plugins", guild_ids=[config.guild_id])
        async def on_chat(ctx, message: str):
            await ctx.defer()
            response = await self.process_message(ctx, message)
            await ctx.respond(response)
            
    def get_functions(self):
        with open("plugins_settings.json", "r") as file:
            plugins = json.load(file)
        enabled_plugins = [plugin for plugin, settings in plugins["plugins"].items() if settings["enabled"]]
        functions = []
        for plugin in enabled_plugins:
            self.plugins.append(plugin)
            with open(f"plugins/{plugin}/functions.json", "r") as file:
                plugin_functions = json.load(file)
                functions.append(plugin_functions)
        combined_functions = []
        for func_list in functions:
            combined_functions.extend(func_list)
        functions = combined_functions
        return functions

    async def process_message(self, ctx, message):
        functions = self.get_functions()
        response = self.get_openai_response(message, functions)
        message = response["choices"][0]["message"]

        if message.get("function_call"):
            function_name = message["function_call"]["name"]
            plugin_folder = self.get_plugin_folder(function_name)
            plugin_class = self.load_plugin_class(plugin_folder)
            response = plugin_class(message, function_name).get_response()
        else:
            response = message["content"]

        return response

    def get_openai_response(self, message, functions):
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo-0613",
            messages=[{"role": "user", "content": message}],
            functions=functions,
            function_call="auto",
        )
        return response

    def get_plugin_folder(self, function_name):
        for plugin in self.plugins:
            plugin_functions_path = Path(f"plugins/{plugin}/functions.json")
            with plugin_functions_path.open("r") as file:
                plugin_functions = json.load(file)
                for function in plugin_functions:
                    if function["name"] == function_name:
                        return plugin

    def load_plugin_class(self, plugin_folder):
        plugin_path = Path(f"plugins/{plugin_folder}/{plugin_folder}.py")
        spec = importlib.util.spec_from_file_location(f"plugins.{plugin_folder}.{plugin_folder}", str(plugin_path))
        plugin_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(plugin_module)
        plugin_folder_parts = plugin_folder.split('_')
        plugin_folder = ''.join([part.capitalize() for part in plugin_folder_parts])
        plugin_class = getattr(plugin_module, plugin_folder)
        return plugin_class
    
    def run(self):
        bot.run(config.bot_token)
        
if __name__ == "__main__":
    DiscordBot()