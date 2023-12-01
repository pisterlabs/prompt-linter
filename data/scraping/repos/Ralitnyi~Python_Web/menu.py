from logo import LOGO

from addressbook import addressbook_app
from file_sorter import sorter_app
from openai_gpt import gpt_app
from notebook import notebook_app
from utilities import completer_input, kb_interrupt_error


def exit_bot():
    print('Чекаю твого найшвидшого повернення!')

def print_menu():
    print(
        f'''
        {LOGO}
                        Menu
                        1. Address book
                        2. Notebook
                        3. File sorter
                        4. Ask gpt
                        0. Exit
        '''
    )

MENU_MAPING = {
    ('1', "Address book"): addressbook_app,
    ('2', "Notebook"): notebook_app,
    ('3', "File sorter"): sorter_app,
    ('4', "Ask gpt"): gpt_app,
    ('0', "Exit"): None,
}

END_COMMANDS = ('0', 'exit', 'close', 'good bye')

menu_commands_list = []

for _, command in MENU_MAPING:
    menu_commands_list.append(command)

@kb_interrupt_error
def main():
    is_working = True

    while is_working:
        print_menu()
        print('Виберіть програму з якою хочете працювати.')
        valid_input = False
        
        while not valid_input:
            user_input = completer_input('>>> ', menu_commands_list)
            for menu_line, app in MENU_MAPING.items():
                if user_input.strip().startswith(menu_line)\
                    and user_input.strip().lower() not in END_COMMANDS:
                    try:
                        app()
                    except SystemExit: # handles sorter app os.exit logic
                        pass
                    valid_input = True
                    break
                elif user_input.strip().lower().startswith(END_COMMANDS):
                    exit_bot()
                    valid_input = True
                    is_working = False
                    break
            else:
                print('Невірно введений пункт меню. Спробуйте ще раз.')


        
if __name__ == '__main__':
    main()
   

