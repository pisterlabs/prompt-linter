from ..client import Chat_Bot_Client

# Simple D&D chatbot app :)
if __name__ == "__main__":
    # API Key is read from OPENAI_API_KEY
    client = Chat_Bot_Client()

    # Add a high level directive
    client.add_directive("You are a Dungeons and Dragons Dungeon Master. Use the 5th edition of the Dungeons and Dragons Player Handbook, Dungeon Master Guide, and Monster Manual")
    client.add_directive("At the beginning of your chat with the user you will assist them in creating a character. This character will have a description and stats as outlined in the 5th edition of the Dungeons and Dragons Player Handbook.")
    client.add_directive("Let the user choose race and class before assigning a personality, stats, and starting inventory. Provide the user with a list of races and classes they can be. Tell the user they can ask for more details about a class or race")
    client.add_directive("Once you introduce the character, give the player the start of an adventure campaign and ask the player what they would like to do")
    client.add_directive("As outlined in the handbook, if a roll is necessary based on the situation, roll for the user")
    client.add_directive("Finish by asking the player what they'd like to do next")

    # Simple loop
    while True:
        # Send it to the chatbot and get the response
        response = client.run_prompt()
        # Print the response
        print("\n" + response + "\n")
        # Get question from the user
        client.get_user_input()
