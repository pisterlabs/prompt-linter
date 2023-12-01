import pyfiglet
from colorama import Fore
import time
import webbrowser
import openai

API_KEY = open("apiKey", "r").read()
openai.api_key = API_KEY

print("\n" + time.asctime() + "\n")

symbols_dictionary = {
    "first_person_symbol": '✘',
    "second_person_symbol": '⬤',
    "blank_space": "□"
}

the_players_want_to_play = True

print(Fore.CYAN + pyfiglet.figlet_format("Welcome!", font="big"))
time.sleep(1.5)

print(Fore.CYAN + pyfiglet.figlet_format("Let's play Tic_Tac_Toe!", font="big"))
time.sleep(1.5)


def win_check(playing_field):
    for row in playing_field:
        if len(set(row)) == 1:
            if row[0] == symbols_dictionary["first_person_symbol"]:
                return "first_person_win"
            elif row[0] == symbols_dictionary["second_person_symbol"]:
                return "second_person_win"

    for index in range(len(playing_field)):
        if len(set([playing_field[index][0], playing_field[index][1], playing_field[index][2]])) == 1:
            if playing_field[index][0] == symbols_dictionary["first_person_symbol"]:
                return "first_person_win"
            elif playing_field[index][0] == symbols_dictionary["second_person_symbol"]:
                return "second_person_win"

    first_diagonal_list = []
    for index in range(len(playing_field)):
        first_diagonal_list.append(playing_field[index][index])
    if len(set(first_diagonal_list)) == 1:
        if first_diagonal_list[0] == symbols_dictionary["first_person_symbol"]:
            return "first_person_win"
        elif first_diagonal_list[0] == symbols_dictionary["second_person_symbol"]:
            return "second_person_win"

    second_diagonal_list = []
    xedni = -1
    for index in range(len(playing_field) - 1, -1, -1):
        xedni += 1
        second_diagonal_list.append(playing_field[index][xedni])
    if len(set(second_diagonal_list)) == 1:
        if second_diagonal_list[0] == symbols_dictionary["first_person_symbol"]:
            return "first_person_win"
        elif second_diagonal_list[0] == symbols_dictionary["second_person_symbol"]:
            return "second_person_win"

    if playing_field[0].count(symbols_dictionary["blank_space"]) == 0 and playing_field[1].count(symbols_dictionary["blank_space"]) == 0 and playing_field[2].count(symbols_dictionary["blank_space"]) == 0:
        print(Fore.YELLOW + pyfiglet.figlet_format("Draw!", font="slant"))
        time.sleep(0.3)
        return "Draw"


def get_bot_move(playing_field):
    prompt = f"We are playing tic-tac-toe. I have placed my piece. (I am playing with this symbol: {symbols_dictionary['first_person_symbol']} and you are playing with this symbol: {symbols_dictionary['second_person_symbol']}. This symbol is an empty space: {symbols_dictionary['blank_space']}) The current game state is: {playing_field}. Choose a row and a position (greater than 0 and smaller than 4) and return a response in the format: 'I choose row (your selected row) position (your selected position)."
    conversation_history = [
        {"role": "user",
         "content": prompt}
    ]

    response = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=conversation_history)

    bot_move = response["choices"][0]["message"]["content"]
    return bot_move


do_you_know_the_rules = str(input(Fore.LIGHTWHITE_EX + "Are you familiar with the game? (Yes/No) "))

if do_you_know_the_rules.lower() == "no":
    time.sleep(0.5)
    print("Two players take turns marking the spaces in a three-by-three grid with X or O. The player who succeeds in placing three of their marks in a horizontal, vertical, or diagonal row is the winner.")
    time.sleep(8)

    more_info = str(input("Do you need more explanation before starting the game? (Yes/No) "))

    if more_info.lower() == "yes":
        time.sleep(0.5)
        webbrowser.open("https://www.youtube.com/watch?v=USEjXNCTvcc")
        time.sleep(15)


while the_players_want_to_play:

    player_vs_player_or_player_vs_bot = str(input("How do you wish to play? Versus a bot or versus another player? (with a bot/with another player) "))

    playing_field = [[symbols_dictionary["blank_space"] for el in range(3)] for row in range(3)]

    for row in playing_field:
        print(Fore.LIGHTWHITE_EX + " ".join(row))
        time.sleep(0.3)

    index = -1
    the_game_is_running = True
    game_over = pyfiglet.figlet_format("Game Over!", font="slant")

    while the_game_is_running:
        index += 1

        if index % 2 == 0:  # First player's turn
            try:
                time.sleep(0.2)
                selected_line = int(input(Fore.CYAN + "Player 1, choose a line: "))
                time.sleep(0.2)
                selected_position = int(input(Fore.CYAN + "Player 1, choose a position: "))
                time.sleep(0.2)

                if selected_position > 3 or selected_position < 1 or selected_line < 1 or selected_line > 3:
                    raise IndexError

                selected_line -= 1
                selected_position -= 1

                selected_position_figure = playing_field[selected_line][selected_position]

                if selected_position_figure != symbols_dictionary["blank_space"]:
                    print(Fore.YELLOW + "This position is already taken! Try again!")
                    index -= 1

                else:
                    playing_field[selected_line][selected_position] = symbols_dictionary["first_person_symbol"]

                    for row in playing_field:
                        print(Fore.CYAN + " ".join(row))
                        time.sleep(0.3)

                    game_check = win_check(playing_field)

                    if game_check == "Draw":
                        print(game_over)
                        the_game_is_running = False

                    elif game_check == "first_person_win":
                        first_player_won = "First player won!"
                        time.sleep(1)
                        print_1 = ""
                        for i in range(0, len(first_player_won)):
                            time.sleep(0.1)
                            print_1 += first_player_won[i]
                            print(Fore.RED + print_1)
                        time.sleep(0.2)
                        print(game_over)
                        the_game_is_running = False

                    elif game_check == "second_person_win":
                        second_player_won = "Second player won!"
                        time.sleep(1)
                        print_2 = ""
                        for i in range(0, len(second_player_won)):
                            time.sleep(0.1)
                            print_2 += second_player_won[i]
                            print(Fore.GREEN + print_2)
                        time.sleep(0.2)
                        print(game_over)
                        the_game_is_running = False

            except:
                print(Fore.YELLOW + "You have entered an invalid position! Try again!")
                index -= 1

        else:  # Second Player's turn
            if player_vs_player_or_player_vs_bot.lower() == "with a bot":  # The opponent is the Bot
                while True:
                    result = get_bot_move(playing_field)
                    time.sleep(0.3)
                    print(Fore.LIGHTWHITE_EX + f'The bot returned: {result}')
                    time.sleep(0.5)
                    result = [int(el) for el in result if el.isdigit()]
                    if len(result) == 2:
                        selected_line = result[0]
                        selected_position = result[1]
                        if selected_line > 0 and selected_line < 4 and selected_position > 0 and selected_position < 4:
                            break
                            # The generated indexes are valid

                    print(Fore.YELLOW + f'The generated response is invalid and a new one will be generated...')

            else:
                try:
                    time.sleep(0.2)
                    selected_line = int(input(Fore.GREEN + "Player 2, choose a line: "))
                    time.sleep(0.2)
                    selected_position = int(input(Fore.GREEN + "Player 2, choose a position: "))
                    time.sleep(0.2)

                    if selected_position > 3 or selected_position < 1 or selected_line < 1 or selected_line > 3:
                        raise IndexError

                except:
                    print(Fore.YELLOW + "You have entered an invalid position! Try again!")
                    index -= 1
                    continue

            selected_line -= 1
            selected_position -= 1

            selected_position_figure = playing_field[selected_line][selected_position]

            if selected_position_figure != symbols_dictionary["blank_space"]:
                if player_vs_player_or_player_vs_bot.lower() == "with a bot":
                    print(Fore.YELLOW + f'The bot\'s entered position is taken and a new one will be generated...')
                else:
                    print(Fore.YELLOW + "This position is already taken! Try again!")
                index -= 1

            else:
                playing_field[selected_line][selected_position] = symbols_dictionary["second_person_symbol"]

                for row in playing_field:
                    print(Fore.CYAN + " ".join(row))
                    time.sleep(0.3)

                game_check = win_check(playing_field)

                if game_check == "Draw":
                    print(game_over)
                    the_game_is_running = False

                elif game_check == "first_person_win":
                    first_player_won = "First player won!"
                    time.sleep(1)
                    print_1 = ""
                    for i in range(0, len(first_player_won)):
                        time.sleep(0.1)
                        print_1 += first_player_won[i]
                        print(Fore.RED + print_1)
                    time.sleep(0.2)
                    print(game_over)
                    the_game_is_running = False

                elif game_check == "second_person_win":
                    second_player_won = "Second player won!"
                    time.sleep(1)
                    print_2 = ""
                    for i in range(0, len(second_player_won)):
                        time.sleep(0.1)
                        print_2 += second_player_won[i]
                        print(Fore.GREEN + print_2)
                    time.sleep(0.2)
                    print(game_over)
                    the_game_is_running = False

    time.sleep(3)
    continuation = str(input("Do you wish to play another game? (Yes/No) "))

    if continuation.lower() != "yes":
        time.sleep(1)
        print(Fore.CYAN + "Thanks for playing!")
        time.sleep(0.5)
        print(Fore.CYAN + "Bye!")
        the_players_want_to_play = False

    elif continuation.lower() == "yes":
        the_players_want_to_play = True
