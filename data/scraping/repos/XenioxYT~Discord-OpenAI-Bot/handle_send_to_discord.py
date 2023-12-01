import discord
from utils.store_conversation import store_conversation
import queue as thread_queue
import threading
import asyncio
import openai
from utils.exponential_backoff import exponential_backoff

def threaded_fetch(response, queue, completion):
    for chunk in response:
        new_content = chunk.choices[0].delta.get("content", "")
        if new_content:
            completion += new_content
            queue.put(new_content)
    print("Response generated from GPT-4!")
    queue.put(None)
    return completion

async def send_to_discord(queue, min_chunk_size, max_chunk_size, delay, temp_message, final_response, message):
    async_queue = asyncio.Queue()
    full_response = ""
    is_first_chunk = True
    embed_message = None  # This will hold the message containing our embed
    i = 0

    def create_embed(content):
        """Helper function to create a discord embed with a standard footer."""
        embed = discord.Embed(description=content)
        embed.set_footer(text="Response converted to embed due to length > 2000 characters.")
        return embed

    while True:
        while not queue.empty():
            item = queue.get()
            await async_queue.put(item)

        try:
            new_content = await asyncio.wait_for(async_queue.get(), timeout=3)
        except asyncio.TimeoutError:
            continue

        if new_content is None:
            break

        full_response += new_content
        final_response += new_content

        # Send the first chunk after 0.3 seconds regardless of its size
        if is_first_chunk:
            print("started to send the first chunk")
            await asyncio.sleep(0.1)
            current_content = full_response
            full_response = ""  # Clear out full_response since we've processed its content
            if temp_message:
                await temp_message.delete()
            try:
                temp_message = await message.channel.send(current_content)
            except discord.errors.HTTPException:
                embed = create_embed(current_content)
                embed_message = await message.channel.send(embed=embed)
            is_first_chunk = False
            i += 1
            continue  # Skip to the next iteration to check for more content
        
        if i == 1:
            min_chunk = 10
        else:
            min_chunk = 50
        
        # Process other chunks when size >= 50
        while len(full_response) >= min_chunk:
            current_content = full_response[:max_chunk_size]
            full_response = full_response[max_chunk_size:]
            i += 1
            try:
                if embed_message:
                    embed_message = await embed_message.channel.fetch_message(embed_message.id)
                    embed = embed_message.embeds[0]
                    embed.description += current_content
                    await embed_message.edit(embed=embed)
                else:
                    temp_message = await temp_message.channel.fetch_message(temp_message.id)
                    await temp_message.edit(content=temp_message.content + current_content)
            except discord.errors.HTTPException:
                if temp_message:
                    await temp_message.delete()  # <-- Delete temp_message if it exists
                if not embed_message:
                    embed = create_embed(temp_message.content + current_content)
                    embed_message = await message.channel.send(embed=embed)
                else:
                    embed_message = await embed_message.channel.fetch_message(embed_message.id)
                    embed = embed_message.embeds[0]
                    embed.description += current_content
                    await embed_message.edit(embed=embed)
            await asyncio.sleep(delay)

    # Processing the last remaining content (if any) in full_response
    if full_response:
        if embed_message:
            embed_message = await embed_message.channel.fetch_message(embed_message.id)
            embed = embed_message.embeds[0]
            embed.description += full_response
            await embed_message.edit(embed=embed)
        else:
            try:
                temp_message = await temp_message.channel.fetch_message(temp_message.id)
                await temp_message.edit(content=temp_message.content + full_response)
            except discord.errors.HTTPException:
                embed = create_embed(full_response)
                await message.channel.send(embed=embed)

    print("Finished with send_to_discord.")
    return final_response, temp_message


async def generate_response(conversation, message, conversation_id):
    response = await exponential_backoff(
        lambda model, latest_conversation: openai.ChatCompletion.create(
        model=model,
        messages=latest_conversation,
        stream=True,
        allow_fallback=True,
        ),
        conversation_id=conversation_id,
        message=message,
    )

    return response

async def update_conversation_and_send_to_discord(function_response, function_name, temp_message, conversation, conversation_id, message):
    conversation.append(
        {
            "role": "function",
            "name": function_name,
            "content": function_response,
        }
    )
    store_conversation(conversation_id, conversation)

    final_response = ""
    try:
        response = await generate_response(conversation, message, conversation_id)
    except Exception as e:
        print(f"Error occurred: {e}")
        temp_message.delete()
        return

    thread_safe_queue = thread_queue.Queue()
    threading.Thread(target=threaded_fetch, args=(response, thread_safe_queue, "")).start()

    completion, temp_message = await send_to_discord(thread_safe_queue, 75, 2000, 0.25, temp_message, final_response, message)

    conversation.append(
        {
            "role": "assistant",
            "content": completion,
        }
    )
    store_conversation(conversation_id, conversation)

    # try:
    #     await temp_message.edit(content=completion)
    # except discord.errors.HTTPException:
    #     embed = discord.Embed(description=completion)
    #     await message.channel.send(embed=embed)
    #     print("embed sent using the second method")
