# Local libs
import oauth_secret
import messages
import utils
import chat

# Built-in libs
import os
import random
from datetime import datetime
import json
import glob

# PyPI libs
import re
import openai
from colorama import Fore
from colorama import Style

os.system('color')

player_file_check = True
skip_menu = False
prev_session = False
log_dir = "Logs"
data_dir = "Data"
log_file_name = ""
i_main = 5
    
dialog = [
            {"role": "system", 
                "content" : messages.initialize}
            ]

ooc_dialog = []

while True:
    if skip_menu == False:
        i_main = 5 # Default menu item -- Number of menu items plus one (for default)
        main_menu = input(
            "1. Start an adventure\n" +
            "2. Play a previous session\n" +
            "3. Adventure Generator\n" +
            "4. General Chat\n" +
            "> "
        )

        if utils.is_integer(main_menu):
            i_main = int(main_menu)
    else:
        skip_menu = False
        
        
    if i_main == 3:       
        with open("d7d_structure_generator.py") as f:
            exec(f.read())
        continue
    elif i_main == 2:
        # Read from previous logfile and attempt to continue the adventure from where it left off.
        if os.path.exists(log_dir + "/") == False:
            os.mkdir(log_dir)
            
        res = []
        n = 0
        files = os.listdir(log_dir)
        #print(files)
        if not files:
            print("DM: There are no previous log files.")
            continue
        else:
            outp = "\nFiles and directories in '" + log_dir + "':\n"
            for path in files:
                #if os.path.isfile(os.path.join(log_dir,path)):
                    res.append(path)
                    outp += "\n" + str(n) +". " + path
                    n+=1
                    
            print(outp)
            read_num = input("DM: Enter the number of the log you want to view or B(b) to go back.\n\nPlayer: ")
            if read_num == "B" or read_num == "b":
                continue
            elif utils.is_integer(read_num) and int(read_num) < len(res) and int(read_num) >= 0:
                #print("DM: Opening " + res[int(read_num)] + "...\n")
                #print(res)
                file = open(log_dir + "/" + res[int(read_num)],"r")
                print("DM: Loaded " + res[int(read_num)] + "!\n\nDo you want to play this adventure? (Y/N)\n\n")
                cont_adv = input("Player: ")
                if cont_adv == "Y" or cont_adv == "y":
                    log_file_name = res[int(read_num)]
                    dialog = json.loads(file.read())
                    #print(dialog)
                    i_main = 1
                    skip_menu = True
                    prev_session = True
                    #print(dialog[-1])
                    if dialog[-1]['role'] == "user":
                        dialog.pop() # remove last user prompt from the list.
                        # print("... Deleted last user prompt\n")
                continue
            else:
                print("DM: That's not a real entry. You should have entered the line number of the file you want to read.")
                continue
        
    elif i_main == 4:
        dialog = [] # blank the array. we don't want it to be a DM, but just a standard ol' chatbot..
        print("\n")
        while chat.active:
            user_msg = input("Player: ")
            if user_msg == "exit":
                break
            else:
                if not dialog:
                    dialog = [
                        {"role":"system",
                            "content":user_msg
                        }
                    ]
                else:
                    dialog.append({"role": "user", "content": user_msg})
                    
                    
                completion = chat.safe_chat_completion(
                    model="gpt-3.5-turbo",
                    messages = dialog
                )
                if completion == -1:
                    continue
                    
                # Capture the AI's response
                assistant_msg = chat.get_content(completion)

                # Append the user input to the ongoing dialog
                dialog.append({"role": "assistant", "content": assistant_msg})

                # Print out chatbot response
                print("\nDM: " + assistant_msg + "\n")
            
    elif i_main == 1:
        full_text_log = ""
        if prev_session == False:
            #Import character sheet
            character_file = input("\nDM: If you have a character sheet file in the Data directory, what is the file name? Leave blank and hit Enter for me to generate a character.\n\nPlayer: ")
            if character_file != "":
                while player_file_check:
                    if os.path.isfile(data_dir + "/" + character_file):    
                        with open(data_dir + "/" + character_file, 'r') as file:
                            character_data = file.read()
                        break
                    else:
                        utils.color_print(f"\n{Fore.RED}Error{Style.RESET_ALL}: Sorry, I couldn't find that file. Try again? Leave blank and hit Enter for me to generate a character.")
                        character_file = input("\nPlayer: ")
                        if character_file != "":
                            continue
                        else:
                            character_data = "The character is random. Please generate its name, race, class and other stats for me."
                            break
            else:
                character_data = "The character is random. Please generate its name, race, class and other stats for me"
                
            dialog.append({"role": "user" , "content":"The players character sheet is as follows:\n\n" + character_data})
               
            # Welcome user and query the for initial chat topic
            print("\nDM: What is the premise of your adventure? Leave blank and hit Enter for me to generate a premise.\n")

            adventure_dialog = [
                                    {"role": "user" , "content": "Come up with a cool premise for a D&D adventure using the following character. " + character_data}
                               ]

            adventure_started = False
        else:
            print("DM: " + dialog[-1]['content'] + "\n\n")
            adventure_started = True
            n = 0
            for history in dialog:
                if history["role"] == "assistant":
                    full_text_log += "DM: " + history["content"] + "\n\n"
                elif history["role"] == "user":
                    full_text_log += "Player: " + history["content"] + "\n\n"
            
            
        
        
        # Enter dialog loop
        while chat.active:

            # Prompt user for there response
            user_msg = input("Player: ")

            # If they didn't enter anything for a premise, then generate one for them.
            if (user_msg == "" and adventure_started == False):
                completion = chat.safe_chat_completion(
                    model="gpt-3.5-turbo",  
                    messages = adventure_dialog
                )
                adventure_premise = chat.get_content(completion)
                dialog.append({"role":"user", "content": adventure_premise})
                
                # Show welcome message and adventure premise.
                utils.color_print(messages.welcome + adventure_premise)
            
            adventure_started = True

            # Check if user wants to exit
            if user_msg == "exit":
                print("\nIt was nice playing D&D with you. Goodbye.\n")
                break
            # Check if user wants to log their adventure then exit
            elif user_msg == "log and exit":
                if log_file_name == "":
                    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
                    
                    # Ask bot for a nice title.
                    dialog.append({"role":"user", "content": "Generate a 8-20 character alphanumeric title for this adventure to be used in a filename."})
                    
                    # Exit with generated title        
                    completion = chat.safe_chat_completion(
                        model="gpt-3.5-turbo",
                        messages = dialog
                    )

                    if completion == -1:
                        assistant_msg = "ExitedWithError"
                    else:
                        assistant_msg = chat.get_content(completion)
                    
                    if os.path.exists(log_dir + "/") == False:
                        os.mkdir(log_dir)
                        
                    title = ''.join(c for c in assistant_msg if c.isalnum())
                    log_file_name = timestamp + "_" + title + ".json"
                
                print("\nDM: Saving log to "+log_file_name)
                file = open(log_dir + "/" + log_file_name,"w")
                file.write(json.dumps(dialog))
                file.close()   
                break
            elif user_msg == "dall-e":
                # Use sparingly! Expensive AND the prompt length max is insanely short.
                if full_text_log == "":
                    print("DM: No prompt has been built yet. Please play a bit before generating an image.\n")
                else:
                    response = openai.Image.create(
                        prompt=full_text_log,
                        n=1,
                        size="512x512"
                    )
                    image_url = response['data'][0]['url']
                    print(f"\nDM: Go to {image_url} to see what I think the current scene looks like.")               
            elif user_msg == "error_test":
                completion = chat.safe_chat_completion(
                    model="gpt-3.5-turbo",
                    messages = "error_test"
                )
                continue
            elif utils.is_ooc(user_msg):
                ooc_table = re.split(r'ooc(\/clear)?\s+',user_msg)
                if ooc_table == "/clear":
                    ooc_dialog = [] # clear the dialog.
                print(user_msg)
                ooc_dialog.append({"role":"user", "content":str(ooc_table[2])})
                completion = chat.safe_chat_completion(
                    model="gpt-3.5-turbo",
                    messages = ooc_dialog
                )
                if completion == -1:
                    continue
                    
                assistant_msg = chat.get_content(completion)
                dialog.append({"role": "assistant", "content": assistant_msg})
                print("\nDM: " + assistant_msg + "\n")
                full_text_log += "DM: " + assistant_msg + "\n\n"
                continue
            elif utils.is_roll(user_msg):		
                # Get the number of dice and the die size
                die = user_msg.split("d")
                
                # If there's not a number before the d, assume it's a 1
                if (die[0]==""):
                    die_count = 1
                else:
                    die_count = int(die[0])
                die_size = int(die[1])
                
                # Build the Dice role string
                rollString = "I rolled " + str(die_count) + " d" + str(die_size) + " for "
                die_total = 0
                for i in range(1, die_count+1):
                    die_value = random.randrange(1, die_size)
                    die_total += die_value # Add to the die total
                    if i == 1:
                        rollString += "a " + str(die_value)
                    elif i == die_count:
                        rollString += ", and a " + str(die_value) + " for a total of " + str(die_total) + "." 
                    else:
                        rollString += ", a " + str(die_value)

                print("\nPlayer: " + rollString)
                
                # Set the user message to the roll results
                user_msg = rollString
                
            # Append the user input to the ongoing dialog
            dialog.append({"role": "user", "content": user_msg + " " + messages.reminder})
            full_text_log += "Player: " + user_msg + "\n\n"

            # Query chatbot
            completion = chat.safe_chat_completion(
                model="gpt-3.5-turbo",
                messages = dialog
            )
            if completion == -1:
                continue
                    
            # Capture the AI's response
            assistant_msg = chat.get_content(completion)

            # Append the user input to the ongoing dialog
            dialog.append({"role": "assistant", "content": assistant_msg})

            # Print out chatbot response
            print("\nDM: " + assistant_msg + "\n")
            full_text_log += "DM: " + assistant_msg + "\n\n"
    else:
        print("Thanks for playing!")
        exit()