from openai import OpenAI
import os



if __name__=="__main__":
	# print(os.environ.get("OPENAI_API_KEY"))

	client = OpenAI(
	    # Defaults to os.environ.get("OPENAI_API_KEY")
	    # api_key=OPENAI_KEY,
	)

	chat_completion = client.chat.completions.create(
	    model="gpt-3.5-turbo",
	    messages=[
	    	{
		    	"role": "system", 
		   		"content": """You are a robot that is able to take decisions in an environment. 
		   		You responses should follow the syntax of the game, which consists of a command and an object (e.g. take book).
		   		This is the list of available commands together with a short description of each command. List:<<<
		   			look:                describe the current room
					goal:                print the goal of this game
					inventory:           print player's inventory
					go <dir>:            move the player north, east, south or west
					examine ...:         examine something more closely
					eat ...:             eat edible food
					open ...:            open a door or a container
					close ...:           close a door or a container
					drop ...:            drop an object on the floor
					take ...:            take an object that is on the floor
					put ... on ...:      place an object on a supporter
					take ... from ...:   take an object from a container or a supporter
					insert ... into ...: place an object into a container
					lock ... with ...:   lock a door or a container with a key
					unlock ... with ...: unlock a door or a container with a key >>> End of List.

		   		The `user` will be the actual game environement telling you about what you see and what you should do."""
	    	},
	    	{
	    		"role": "user",
	    		"content": """Welcome to TextWorld! Here is how to play! First, it would be a great idea if you could pick up the non-euclidean passkey from the floor of the attic. And then, unlock the non-euclidean locker. Having unlocked the non-euclidean locker, ensure that the non-euclidean locker within the attic is open. After that, pick up the type 4 key from the non-euclidean locker in the attic. And then, lock the type 4 chest with the type 4 key. That's it!"""
	    	},
	    	{
	    		"role": "user",
	    		"content": """You are in an attic. A standard kind of place. The room is well lit.

You smell an intriguing smell, and follow it to a type 4 chest. There's something strange about this being here, but you can't put your finger on it. Were you looking for a non-euclidean locker? Because look over there, it's a non-euclidean locker. Now that's what I call TextWorld! You can make out a shelf. But oh no! there's nothing on this piece of trash.

You need an unguarded exit? You should try going west.

There is a non-euclidean passkey and a lightbulb on the floor."""
	    	}
	    ]
	)

	# print(chat_completion)
	# print(chat_completion.choices[0].message)
	print(chat_completion.choices[0].message.content)
	print(chat_completion.choices[0].message.role)