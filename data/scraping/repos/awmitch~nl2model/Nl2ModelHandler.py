import discord
import asyncio
from typing import Dict, List, Any, Optional, Union
from uuid import UUID
from langchain.callbacks.base import AsyncCallbackHandler
from discord import TextChannel
from langchain.schema import (
    AgentAction,
    AgentFinish,
    LLMResult,
)
import json
from discord import File
class DiscordHandler(AsyncCallbackHandler):
    discord_channel: TextChannel = None
    current_message: discord.Message = None
    lock: asyncio.Lock = asyncio.Lock()
    
    def set_channel(self, channel: TextChannel):
        self.discord_channel = channel

    async def on_results(self,plot_buffer):
        await self.discord_channel.send(file=File(fp=plot_buffer, filename='plot.png'))

    async def update_message(self, new_content: str):
        async with self.lock:
            if len(new_content) > 2000:  # handle cases where new_content itself exceeds Discord limit
                for chunk in self.split_long_message(new_content):
                    await self.update_message(chunk)
            else:
                if self.current_message is None:
                    self.current_message = await self.discord_channel.send(new_content)
                else:
                    current_content = self.current_message.content
                    updated_content = current_content + "\n" + new_content
                    if len(updated_content) > 2000:  # if update exceeds Discord limit, send a new message
                        self.current_message = await self.discord_channel.send(new_content)
                    else:
                        await self.current_message.edit(content=updated_content)
                        self.current_message = await self.discord_channel.fetch_message(self.current_message.id)
    
    def split_long_message(self, message: str) -> List[str]:
        """Splits a long message into chunks each of length less than or equal to 2000 characters."""
        return [message[i: i + 2000] for i in range(0, len(message), 2000)]
    
    def format_as_markdown(self,data, level=1):
        if isinstance(data, dict):
            result = []
            for key, value in data.items():
                if level == 1:
                    result.append(f"`{key}`")
                elif level == 2:
                    result.append(f"*{key}*")
                if isinstance(value, dict):
                    result.append(self.format_as_markdown(value, level + 1))
                else:
                    result.append(f"```\n{value}\n```")
            return "\n".join(result)
        else:
            return f"```\n{data}\n```"
        
    
    async def on_chain_start(self, serialized: Dict[str, Any], inputs: Dict[str, Any], **kwargs: Any) -> None:
        chain_name = serialized.get('name', 'Unnamed Chain')
        message = f"**{chain_name}**\n{self.format_as_markdown(inputs)}"
        await self.update_message(message)

    async def on_llm_start(self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any) -> None:
        llm_name = serialized.get('name', 'Unnamed LLM')
        message = f"**{llm_name}**\n{self.format_as_markdown(prompts)}"
        await self.update_message(message)

    async def on_llm_new_token(self, token: str, **kwargs) -> None:
        message = f"{token}"
        await self.update_message(message)

    async def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        message = f"**LLM End**\n{self.format_as_markdown(response)}"
        await self.update_message(message)

    async def on_chain_end(self, outputs: Dict[str, Any], **kwargs: Any) -> None:
        message = f"**Chain End**\n{self.format_as_markdown(outputs)}"
        await self.update_message(message)

    async def on_agent_action(self, action: AgentAction, **kwargs) -> None:
        message = f"**Agent Action**\n{self.format_as_markdown(action)}"
        await self.update_message(message)

    async def on_tool_start(self, serialized: Dict[str, Any], input_str: str, **kwargs: Any) -> None:
        tool_name = serialized.get('name', 'Unnamed Tool')
        message = f"**{tool_name}**\n{self.format_as_markdown(input_str)}"
        await self.update_message(message)

    async def on_tool_end(self, output: str, **kwargs: Any) -> None:
        message = f"**Tool End**\n{self.format_as_markdown(output)}"
        await self.update_message(message)

    async def on_agent_finish(self, finish: AgentFinish, **kwargs: Any) -> None:
        message = f"**Agent Finished**\n{self.format_as_markdown(finish)}"
        await self.update_message(message)
    # Handle errors
    async def on_llm_error(self, error: Union[Exception, KeyboardInterrupt], **kwargs: Any) -> None:
        message = f"**LLM Error**\n{str(error)}"
        await self.update_message(message)

    async def on_chain_error(self, error: Union[Exception, KeyboardInterrupt], **kwargs: Any) -> None:
        message = f"**Chain Error**\n{str(error)}"
        await self.update_message(message)

    # Handle errors
    async def on_tool_error(self, error: Union[Exception, KeyboardInterrupt], **kwargs: Any) -> None:
        message = f"**Tool Error**\n{str(error)}"
        await self.update_message(message)



class Nl2ModelHandler(DiscordHandler):
    async def on_chain_start(self, serialized: Dict[str, Any], inputs: Dict[str, Any], **kwargs: Any) -> None:
        pass
    async def on_llm_start(self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any) -> None:
        pass
    async def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        pass
    async def on_agent_action(self, action: AgentAction, **kwargs) -> None:
        pass
    async def on_chain_end(self, outputs: Dict[str, Any], **kwargs: Any) -> None:
        pass

class LookupHandler(DiscordHandler):
    async def on_llm_start(self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any) -> None:
        pass
    async def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        pass
class ResponseHandler(DiscordHandler):
    pass
class ModelicaHandler(DiscordHandler):
    def on_chain_start(self, serialized: Dict[str, Any], inputs: Dict[str, Any], **kwargs: Any) -> None:
        return asyncio.run(self.on_chain_start(serialized,inputs, **kwargs))
    def on_chain_start(self, serialized: Dict[str, Any], inputs: Dict[str, Any], **kwargs: Any) -> None:
        chain_name = serialized.get('name', 'Unnamed Chain')
        message = f"**Chain: {chain_name}**\n{self.format_as_markdown(inputs)}"
        asyncio.run(self.update_message(message))

    def on_llm_start(self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any) -> None:
        pass

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        pass
    def on_chain_end(self, outputs: Dict[str, Any], **kwargs: Any) -> None:
        message = f"**Chain End**\n{self.format_as_markdown(outputs)}"
        asyncio.run(self.update_message(message))

    # Handle errors
    async def on_llm_error(self, error: Union[Exception, KeyboardInterrupt], **kwargs: Any) -> None:
        message = f"**LLM Error**\n{str(error)}"
        asyncio.run(self.update_message(message))

    async def on_chain_error(self, error: Union[Exception, KeyboardInterrupt], **kwargs: Any) -> None:
        message = f"**Chain Error**\n{str(error)}"
        asyncio.run(self.update_message(message))

    # Handle errors
    async def on_tool_error(self, error: Union[Exception, KeyboardInterrupt], **kwargs: Any) -> None:
        message = f"**Tool Error**\n{str(error)}"
        asyncio.run(self.update_message(message))




class SummaryHandler(DiscordHandler):
    pass
class QuestionHandler(DiscordHandler):
    pass
class CompressorHandler(DiscordHandler):
    async def on_chain_start(self, serialized: Dict[str, Any], inputs: Dict[str, Any], **kwargs: Any) -> None:
        pass
    async def on_llm_start(self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any) -> None:
        pass
    async def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        pass
    async def on_chain_end(self, outputs: Dict[str, Any], **kwargs: Any) -> None:
        pass