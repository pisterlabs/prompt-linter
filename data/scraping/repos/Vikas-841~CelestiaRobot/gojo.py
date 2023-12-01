import openai
from pyrogram import filters
from Celestia import Celestia


openai.api_key = "sk-ATPjDUOvAGgl9NLygl39T3BlbkFJHxn1gsew2Zv8Zz0rSMdJh"


def generate_response(prompt):
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=prompt,
        max_tokens=150,
        temperature=0.7,
        n=1,
    )
    return response.choices[0].text.strip()


prompt = "You are Gojo Satoru, responding to someone who just challenged you to a fight. What do you say?"

@Celestia.on_message(filters.command("gojo"))
async def gojo(_,message):
    query = message.text.split(maxsplit=1)[1]
  
    prompt += f"\nGojo Satoru: {query}"
    response = generate_response(prompt)
    gojo_response = response[len(prompt):].strip()
    await message.reply_text(gojo_response)


