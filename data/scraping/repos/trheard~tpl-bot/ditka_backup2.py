import discord
import openai

intents = discord.Intents.default()
intents.messages = True

client = discord.Client(intents=intents)
openai_api_key = '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'  # Replace with your OpenAI API key

openai.api_key = openai_api_key

@client.event
async def on_ready():
    print(f'{client.user} has connected to Discord!')

    channel_id = 1135707302775767110  # Replace with your channel ID
    channel = client.get_channel(channel_id)

    # Check send message perms
    print(channel.permissions_for(channel.guild.me).send_messages)

@client.event
async def on_message(message):
    # Skip messages from the bot itself
    if message.author == client.user:
        return

    print(f"Received message in channel {message.channel.id} from {message.author}")

    if message.content.startswith('!askDitka'):
        print("Received askDitka command")
        user_question = message.content[len('!askDitka '):].strip()
        print(f"User question: {user_question}")

        prompt = f"Imagine you are an angry Coach Mike Ditka responding to the following question: {user_question}\n\nResponse:"
        print(f"Sending prompt to OpenAI: {prompt}")

        try:
            response = openai.Completion.create(
              engine="gpt-3.5-turbo",  # Use GPT-3.5-turbo engine
              prompt=prompt,
              max_tokens=100
            )
            ditka_response = response.choices[0].text
            print(f"Generated response from OpenAI: {ditka_response}")

            await message.channel.send(f"Coach Ditka says: {ditka_response}")
            print("Message sent successfully")

        except Exception as e:
            print(f"Error: {e}")


client.run('MTEzNzE5OTg4OTk5OTIwODQ0OA.GU86xE.W6eGJXDGEUT5PIfzxdyb-cikSR8YW_XOmRkzQk')  # Replace with your Discord bot token

