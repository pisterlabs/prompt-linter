from cmd import Cmd
import openai 
import os
import shutil
# from dataclasses import dataclass
import textwrap
import pickle
from colorama import Fore as F, Style as S

cols, rows = shutil.get_terminal_size()

BOLD = '\033[1m'
NORM = '\033[0m'
REV = '\033[7m'

CLEAR = 'cls' if os.name == 'nt' else 'clear' if os.name == 'posix' else 'c'
if CLEAR == 'c': exit()

# ----------------------------------- Reading Sys Info -------------------------------- 

if os.name == 'posix': DATAPATH = f"/home/{os.getlogin()}/.custom_script_data/Wizard.bin"
else: DATAPATH = f"C:/Users/{os.getlogin()}/.custom_script_data/Wizard.bin"
PATH = DATAPATH[:-10]

try:
    with open(DATAPATH,'rb+') as f:
        SYS_INFO = pickle.load(f)
except:
    print("\n\nX Error in Reading System Settings")
    exit()

class GLOBALSTATE:
    realstate = []
    loaded = None
    messages = []

# -------------------------------------------------------------------------------------

os.system(CLEAR)

BANNER = []
Banner = '\n\n' + ('\n'.join([i.center(cols+10," ")[:-10] for i in BANNER])) + '\n\n' + f"{BOLD + S.BRIGHT + F.CYAN}Wizard {F.RED}NEBULA{F.RESET}, {F.LIGHTBLUE_EX}Beholder{F.RESET} of the {F.CYAN}Answers".center(cols+40," ")

USERNAME = os.getlogin().upper()

CHATS_PATH = PATH + "Wiz_Chats/"


WRAPPEROBJECT = textwrap.TextWrapper(width=cols-5)
def wrap(text):
    lines = text.split('\n')
    res = []
    for line in lines:
        res.append('\n  '.join(WRAPPEROBJECT.wrap(line)))
    return '\n  '.join(res)

def TAB_COMPLETER_CHATNAMES(self, text, line, begidx, endidx):
    if text: return [i for i in SYS_INFO["CHATS"] if i.startswith(text)]
    else: return list(SYS_INFO["CHATS"])

class Interactor(Cmd):

    intro = Banner
    prompt = f" \n{F.RED+REV+BOLD} {USERNAME} {NORM}{BOLD} >{F.GREEN} \n\n"+'  '

    def __init__(self):
        super().__init__()
        self.model = "gpt-3.5-turbo"
        self.mode = "chat"
        openai.api_key = SYS_INFO['API_KEY']

    def do_cls(self, args):
        """Clears the screen"""
        os.system(CLEAR)
        print(self.intro)
    do_clear = do_cls

    def do_cload(self, args):
        """Loads a chat session"""
        args = args.split()
        try:
            if not args:
                if not len(os.listdir(CHATS_PATH)): print(f"\n{F.RED}  No Chats Saved"); return
                print(f"\n{F.CYAN} ",wrap("  ".join(list_chats()))+'\n')
                name = input(f"{F.WHITE}  Which chat to load : ").lower().replace(" ", "_")
            else: name = args[0].lower().replace(" ", "_")
        except EOFError as e: print(f"\n\n{F.RED}  Chat could not be Loaded"); return
        except KeyboardInterrupt as e: print(f"\n{F.RED}  Chat could not be Loaded"); return
        try:
            with open(PATH+f"Wiz_Chats/{name}.bin", "rb+") as f:
                GLOBALSTATE.messages = pickle.load(f)
                GLOBALSTATE.loaded = name
                print("> Messages loaded Successfully")
        except Exception as e: print(f"\n{F.RED}X File not Found"); return
        os.system(CLEAR)
        print(self.intro)
        for message in GLOBALSTATE.messages:
            if message["role"] == "system": continue
            elif message["role"] == "user":
                print(f" \n{F.RED+REV+BOLD} {USERNAME} {NORM}{BOLD} >{F.GREEN} \n\n"+'  ')
                print("",wrap(message["content"]))
            else:
                print("\n"+f"{F.LIGHTBLUE_EX + REV + BOLD} NEBULA {NORM + BOLD} >\n\n  "+ wrap(message["content"]))
    complete_cload = TAB_COMPLETER_CHATNAMES

    def do_csave(self, args):
        """Saves the current chat session"""
        args = args.split()
        if args: name = args[0]
        elif GLOBALSTATE.loaded is None: name = take_name()
        else: name = GLOBALSTATE.loaded
        save_chat(name, GLOBALSTATE.messages)

    def do_crm(self, args):
        """Removes a chat session"""
        args = args.split()
        try :
            if args: name = args[0]
            else: 
                if not len(os.listdir(CHATS_PATH)):  print(f"\n{F.RED}  No Chats Saved"); return
                print(f"\n{F.CYAN} ",wrap("  ".join(list_chats()))+'\n')
                name = input(f"{F.WHITE}?  Which chat to remove : ")
            if input(f"\n{F.YELLOW}? Are you sure want to remove {F.CYAN}{name}{F.YELLOW}? [y/n] : ").lower() not in 'yes': print(f"\n{F.GREEN}  Cancelled")
        except Exception: print(f"{F.RED} Error Occured"); return 
        try:
            os.remove(PATH+f"Wiz_Chats/{name}.bin")
            print(f"\n{F.RED}  Chat session removed successfully")
        except FileNotFoundError as e: print(f"\n{F.RED}  Chat session does not exist")
        except KeyError as e: print(f"{F.RED}X Chat could not be removed")
    complete_crm = TAB_COMPLETER_CHATNAMES

    def default(self, line):
        """Chats with OpenAi models"""
        if line == "EOF": raise KeyboardInterrupt()
        if self.mode == "chat":
            GLOBALSTATE.messages.append({"role": "user", "content": line})
            try:
                chat = openai.ChatCompletion.create(
                    model=self.model, messages = GLOBALSTATE.messages
                )
            except openai.error.APIConnectionError as e:
                GLOBALSTATE.messages.pop()
                print(F.RED+"\nX Connection Could not be Established"); return
            except openai.error.RateLimitError as e:
                GLOBALSTATE.messages.pop()
                print(F.RED + f"\n X Rate Limit Exceeded : {F.CYAN}Consider upgrading your API key"); return
            except openai.error.InvalidRequestError as e:
                if str(e).startswith("This model's maximum context length is"):
                    for _ in range(4):
                        GLOBALSTATE.messages.pop(1)
                        GLOBALSTATE.messages.pop(1)
                GLOBALSTATE.messages.pop()
                self.default(line)
            except KeyboardInterrupt: 
                GLOBALSTATE.messages.pop()
                return 


            reply = wrap(chat.choices[0].message.content.lstrip())
            GLOBALSTATE.messages.append({"role": "assistant", "content": reply})
            print("\n"+f"{F.LIGHTBLUE_EX + REV + BOLD} NEBULA {NORM + BOLD} >\n\n  "+ reply)
    
    def do_exit(self, args):"""Exits the Tool"""; raise KeyboardInterrupt()

# --------------------------------------- Util Functions --------------------------------------

def save_chat(name, messages):
    SYS_INFO['CHATS'].add(name)
    try:
        with open(PATH+f"Wizard.bin","wb+") as f:
            pickle.dump(SYS_INFO, f)
        with open(PATH+f"Wiz_Chats/{name}.bin","wb+") as f:
            pickle.dump(messages,f)
            print(f"\n{F.YELLOW}  Chat saved Successfully.\n")
    except Exception as e: print(f"\n{F.RED}  Chat could not be saved, Retry\n")

def take_name():
    name = input(f"\n {F.BLUE}> Name for the chat (Max 50 chars): ").strip().lower().replace(' ', '_')
    while name == "" or len(name)>50 and os.path.exists(PATH+f"Wiz_Chats/{name}.bin"):
        name = input("\nName already exists. Write a valid name : ").strip().lower().replace(' ', '_')
    return name

def list_chats():
    chats = os.listdir(CHATS_PATH)
    return [x[:-4] for x in chats]

# ------------------------------------------ Main ------------------------------------------------

def main():
    GLOBALSTATE.messages = [{"role":"system","content":"Your are Doby. You extensively use emojis in your messages to make them more expressive."}]

    try:
        interactor = Interactor()
        interactor.cmdloop()
    except KeyboardInterrupt as e:
        try:
            if GLOBALSTATE.loaded is not None:
                save_chat(GLOBALSTATE.loaded, GLOBALSTATE.messages)
                print(f'Chat Saved Successfully : {GLOBALSTATE.loaded}')
            elif len(GLOBALSTATE.messages) >= 3:
                save = input(f"\n\n {F.RED}> Do you want to save the chat? [y/n] : ").lower()
                if save in 'yes':
                    name = take_name()
                    save_chat(name, GLOBALSTATE.messages)
        except EOFError as e: pass
        except KeyboardInterrupt as e: pass
        os.system(CLEAR)
    except ModuleNotFoundError as e:
        print(F.RED+f"\n X Module Not Found : {F.CYAN}Consider pip installing unavaillable modules")

if __name__ == '__main__': main()
