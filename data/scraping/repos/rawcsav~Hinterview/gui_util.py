import os
from colorama import Fore, Style
from art import *
from config import configure_settings,get_config
from openai_util import embed_documents

def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

def display_intro():
    clear_screen()

    # Generate the ASCII art text with 'slant' font
    ascii_art = text2art("Hinterview", "slant")

    # Print the ANSI escape codes for bright cyan color
    print(Style.BRIGHT + Fore.CYAN, end="")

    # Replace both the '/' and '_' characters with the desired colors
    colored_ascii_art = ascii_art.replace("/", Fore.GREEN + "/" + Fore.CYAN)
    colored_ascii_art = colored_ascii_art.replace("_", Fore.GREEN + "_" + Fore.CYAN)

    # Print the generated ASCII art with the desired colors
    print(colored_ascii_art)
    print(Fore.CYAN + "──────────────────────────────────────────────────────────────────────────")
    configure_settings()
    folder_path = get_config('folder_path')
    print("\nCurrent directory path:" + Fore.LIGHTGREEN_EX + Style.BRIGHT + f"{folder_path}\n")



def display_initial_menu():
    print(Fore.YELLOW + "1. Continue to Program")
    print(Fore.YELLOW + "2. Open Settings Menu")
    choice = input(Fore.GREEN + "Please select an option (1-2): ")
    return choice

def display_settings_menu():
    clear_screen()
    print(Fore.CYAN + "──────────────────────────────────────────────────────────────────────────")
    print(Style.BRIGHT + Fore.GREEN + "                          SETTINGS")
    print(Fore.YELLOW + "1. Folder Path")
    print(Fore.YELLOW + "2. OpenAI API Key")
    print(Fore.YELLOW + "3. Hotkey")
    print(Fore.YELLOW + "4. Interview Mode")
    print(Fore.YELLOW + "5. GPT Model")
    print(Fore.YELLOW + "6. System Prompt")
    print(Fore.YELLOW + "7. Temperature")
    print(Fore.YELLOW + "8. Max Tokens")
    print(Fore.YELLOW + "9. Resume Title")
    print(Fore.YELLOW + "10. Job Description Title")
    print(Fore.CYAN + "──────────────────────────────────────────────────────────────────────────")
    print(Fore.GREEN + "0. Return to Main Menu")
    choice = input(Fore.LIGHTGREEN_EX + "Please select an option (0-10): ")
    return choice

def handle_settings_menu():
    while True:
        choice = display_settings_menu()
        if choice == '0':
            display_intro()
            break
        elif choice in ('1', '2', '3', '4', '5', '6', '7', '8', '9', '10'):
            settings_options = {
                '1': ('Enter the new folder path: ', 'folder_path'),
                '2': ('Enter the new OpenAI API Key: ', 'openai_api_key'),
                '3': ('Enter the new hotkey: ', 'hotkey'),
                '4': ('Enter the new special option value: ', 'special_option'),
                '5': ('Enter the new GPT model: ', 'gpt_model'),
                '6': ('Enter the new system prompt: ', 'system_prompt'),
                '7': ('Enter the new temperature value: ', 'temperature'),
                '8': ('Enter the new max tokens value: ', 'max_tokens'),
                '9': ('Enter the new resume title: ', 'resume_title'),
                '10': ('Enter the new job description title: ', 'job_description_title'),
            }
            prompt, setting_name = settings_options[choice]
            new_value = input(Fore.GREEN + prompt)
            configure_settings(**{setting_name: new_value})
            print(Fore.GREEN + "Setting updated successfully!")
            clear_screen()
        else:
            print(Fore.RED + "Invalid choice. Please try again.")

def display_instructions():
    print("\nPress and hold the hotkey (default: Option) to record a segment of your interview.")
    print("Release the key to stop recording and get insights.")


def display_recording():
    print(Fore.CYAN + "\n──────────────────────────────────────────────────────────────────────────")
    print(Fore.YELLOW + "\n[STATUS] Recording...")


def display_transcribing():
    print(Fore.BLUE + "[STATUS] Transcribing...")


def display_processing():
    print(Fore.MAGENTA + "[STATUS] Fetching AI Response...")


def display_error(error_message):
    print(Fore.CYAN + "\n──────────────────────────────────────────────────────────────────────────")
    print(Fore.RED + "\nError:", error_message)

def primary_gui():
    display_intro()

    while True:
        choice = display_initial_menu()

        if choice == '1':
            print(Fore.GREEN + "Continuing to the Program...\n")
            break
        elif choice == '2':
            handle_settings_menu()
        else:
            print(Fore.RED + "Invalid choice. Please try again.")
    FOLDER_PATH = get_config("folder_path")
    df = embed_documents(FOLDER_PATH)

    display_instructions()

    return df
