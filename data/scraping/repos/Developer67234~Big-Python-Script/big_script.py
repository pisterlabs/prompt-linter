# start of the script
print("Welcome to big_script.py!")
print("*************************")

# ask user for username
username = input("Please enter your username to continue: ")

if username == "Python_User123":
    print("Welcome back, Python_User123")
else:
    print("Error! Username Incorrect, Goodbye")
    exit()

# ask user for password
password = input("Please enter your password to continue: ")

if password == "Password":
    print("Access Granted!")
    input("Press ENTER to continue")
else:
    print("Error! Password Incorrect, Goodbye")
    exit()

# ask user for their first name
firstname = input("Hello, what is your first name? ")

# Remove whitespace from str
firstname = firstname.strip()

# capitalize user's first name
firstname = firstname.title()

# ask user for their  last name
lastname = input("Okay, what is your last name? ")

# Remove whitespace from str
lastname = lastname.strip()

# capitalize user's last name
lastname = lastname.title()

# tell his/her full name
print ("Welcome back, Mr.",lastname)

# says "nice to meet you" message
print("Wasn't that cool? Nice to meet you!")

# ask for opinion on the python script
opinion = input("What do you think of this Python script? ")

# says thanks for the opinion
print(f"I'm glad to hear that, {firstname}.")
print(" ")
print("I will now guide you through a series of tools I have created, starting off with a functional calculator!")

# says we'll do addition problem
print("Now, lets do an addition problem using x and y.")

# asks for values of x and y
x = float(input("Give me a numerical value of x: "))

y = float(input("Now, give me a numerical value of y: "))

z = x + y

print(x,"+",y,"=")
print(f"{z:.2f}")
print("⬆️  - Answer above")

# says we'll do a subtration problem
print("That was cool!, Lets do a subtraction problem using x and y.")

# asks for values of x and y
x = float(input("Give me a numerical value of x: "))

y = float(input("Now, give me a numerical value of y: "))

z = x - y

print(x,"-",y,"=")
print(f"{z:.2f}")
print("⬆️  - Answer above")

# says we'll do a multiplication problem
print("Let's do a multiplication problem using x and y.")

# asks for values of x and y
x = float(input("Give me a numerical value of x: "))

y = float(input("Now, give me a numerical value of y: "))

z = int(x) * y

print(x,"x",y,"=")
print(f"{z:.2f}")
print("⬆️  - Answer above")
print(" ")


# says we'll do a division problem
print("Finally, lets do a division problem using x and y.")

# asks for values of x and y
x = float(input("Give me a numerical value of x: "))

y = float(input("Now, give me a numerical value of y: "))

z = x / y

print(x,"÷",y,"=")
print(f"{z:.2f}")
print("⬆️  - Answer above")
print(" ")

print("Ok then, the next tool is a UAE quiz!")
print(" ")

# uae quiz
score = 0
print("Welcome to the UAE quiz!")
print("There will be 7 questions which will challenge your understanding about the UAE!")
print("If you get one question wrong, the game ends! Good Luck :)")

# question 1
print(" ")
question1answer = input("What is the tallest building in the UAE? ")

if question1answer == "Burj Khalifa":
     print("Correct!")
     print(" ")
     score = score+1
     print("Your score is",score)
  
else:
  print("Wrong!")
  print(" ")
  score = score-1
  print("Your score is",score)

# question 2
print(" ")
question2answer = input("When was the UAE founded? ")

if question2answer == "1971":
     print("Correct!")
     print(" ")
     score = score+1
     print("Your score is",score)
  
else:
  print("Wrong!")
  print(" ")
  score = score-1
  print("Your score is",score)
  
# question 3
print(" ")
question3answer = input("Which emirate joined the UAE in 1972? ")

if question3answer == "Ras Al Khaimah":
     print("Correct!")
     print(" ")
     score = score+1
     print("Your score is",score)
     
else:
  print("Wrong!")
  print(" ")
  score = score-1
  print("Your score is",score)

# question 4
print(" ")
question4answer = input("Who was the ruler of Dubai? ")

if question4answer == "Sheikh Mohammed":
     print("Correct!")
     print(" ")
     score = score+1
     print("Your score is",score)
     
else:
  print("Wrong!")
  print(" ")
  score = score-1
  print("Your score is",score)

# question 5
print(" ")
question5answer = input("Who is the ruler of the UAE (very hard)? ")

if question5answer == "Sheikh Zayed":
     print("Correct!")
     print(" ")
     score = score+1
     print("Your score is",score)
  
else:
  print("Wrong!")
  print(" ")
  score = score-1
  print("Your score is",score)

# question 6
print(" ")
question6answer = input("Which region is the UAE located in? ")

if question6answer == "Middle East":
     print("Correct!")
     print(" ")
     score = score+1
     print("Your score is",score)
     print(" ")
     print("Be careful though, while there are greater rewards for this question, there are also greater risks!")
else:
  print("Wrong!")
  print(" ")
  score = score-1
  print("Your score is",score)
  print("Be careful though, while there are greater rewards for this question, there are also greater risks!")
  
# question 7
print(" ")
question7answer = input("What is the average UAE population as of 2021? (Double Points)!")


if question7answer == "9 million":
    score = score+2

else:
    score = score-2

# quiz ending
if score>0:
     firstname = firstname+","
     print("You,",firstname,"with a final score of",score,)
     print("Has just won the UAE quiz as you are the only one playing it right now!")
     print(" ")
  
else:
   firstname = firstname+","
   print("Unfortunately you,",firstname,"with a total score of just",score,)
   print("Has just failed the UAE quiz and this script must be stoped")
   print(" ")
   input = ("Do you,"),firstname("have any last words?")
   print(" ")
   print("Well, whatever it is, I'm not reading it. Bye 👋")
   
   exit()

print("Now, let's test out the sleep calculator!")
print(" ")

# sleep calculator
hourspernight = input("How many hours per night do you sleep? ")
hoursperweek = int(hourspernight) * 7
print ("You sleep",hoursperweek,"hours per week")

hourspermonth = float(hoursperweek) * 4.35
print ("You sleep",hourspermonth,"hours per month")

dayspermonth = int(hourspermonth) / 24
print ("You have slept for",dayspermonth,"days a month")
print(" ")

# tic tac toe
print("Now that we know how much you sleep, let's relax with a classic game!")
print("Let's play a game of Tic-Tac-Toe!")

from enum import Enum
import random

class OccupiedBy(Enum):
    NONE = 0
    PLAYER = 1
    COMPUTER = 2

class Winner(Enum):
    NONE = 0
    PLAYER = 1
    COMPUTER = 2
    DRAW = 3

def render_board(board, space_mapping):
    # Render the Tic-Tac-Toe board
    print(" " + space_mapping[board[0]] + " | " + space_mapping[board[1]] + " | " + space_mapping[board[2]])
    print("---+---+---")
    print(" " + space_mapping[board[3]] + " | " + space_mapping[board[4]] + " | " + space_mapping[board[5]])
    print("---+---+---")
    print(" " + space_mapping[board[6]] + " | " + space_mapping[board[7]] + " | " + space_mapping[board[8]])

def determine_winner(board, current_player):
    # Determine the winner of the Tic-Tac-Toe game
    winning_combinations = [
        [0, 1, 2], [3, 4, 5], [6, 7, 8],  # Rows
        [0, 3, 6], [1, 4, 7], [2, 5, 8],  # Columns
        [0, 4, 8], [2, 4, 6]  # Diagonals
    ]

    for combination in winning_combinations:
        if board[combination[0]] == board[combination[1]] == board[combination[2]] == current_player:
            if current_player == OccupiedBy.PLAYER:
                return Winner.PLAYER
            else:
                return Winner.COMPUTER

    if OccupiedBy.NONE not in board:
        return Winner.DRAW

    return Winner.NONE

def computer_think(board):
    # Implement the logic for the computer's move
    available_moves = [index for index, cell in enumerate(board) if cell == OccupiedBy.NONE]
    
    # Check if there are available moves
    if not available_moves:
        return None

    return random.choice(available_moves)

def prompt_player(board):
    while True:
        move = int(input("\nWhere do you move? "))

        if move == 0:
            return 0

        if move > 9 or board[move - 1] is not OccupiedBy.NONE:
            print("That square is occupied :(. Please choose another square.\n\n")
            continue

        return move

def main():
    board = [OccupiedBy.NONE] * 9
    current_player = OccupiedBy.PLAYER
    space_mapping = {
        OccupiedBy.NONE: " ",
        OccupiedBy.PLAYER: "X",
        OccupiedBy.COMPUTER: "O",
    }

    print("Now that we know how much you sleep, let's relax with a classic game!")
    print("Let's play a game of Tic-Tac-Toe!")
    print("\n" + " " * 30 + "TIC-TAC-TOE")
    print(" " * 15 + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY")
    print("\n\n")
    print("The board is numbered:")
    print(" 1  | 2  | 3 ")
    print("----+----+----")
    print(" 4  | 5  | 6 ")
    print("----+----+----")
    print(" 7  | 8  | 9 ")
    print("\n\n")

    symbol = input("Do you want to be 'X' or 'O'? ").upper()

    if symbol != "X":
        space_mapping[OccupiedBy.PLAYER] = "O"
        space_mapping[OccupiedBy.COMPUTER] = "X"
        current_player = OccupiedBy.COMPUTER

    while True:
        if current_player == OccupiedBy.PLAYER:
            move = prompt_player(board)
            if move == 0:
                print("Thanks for the game :)")
                break
        else:
            print("The computer is thinking...")
            move = computer_think(board)
            if move is None:
                print("No available moves for the computer. It's a draw!")
                break

        board[move - 1] = current_player

        render_board(board, space_mapping)

        winner = determine_winner(board, current_player)

        if winner != Winner.NONE:
            if winner == Winner.PLAYER:
                print("Congratulations! You won! GG! :)")
            elif winner == Winner.COMPUTER:
                print("The computer wins! GG! :)")
            else:
                print("It's a draw!")
            break

        if current_player == OccupiedBy.PLAYER:
            current_player = OccupiedBy.COMPUTER
        else:
            current_player = OccupiedBy.PLAYER

if __name__ == "__main__":
    main()
    
print(" ")
print("Now that we're done with Tic-Tac-Toe, it's now time for you,",firstname,)
print("To play a game of Hangman!")

import random

words = ["apple", "banana", "cherry", "date", "elderberry", "helicopter", "medicine", "delicious","scrumptious","devour","eloquent","orange","maroon","magenta","chapters"]
word = random.choice(words)
guessed_letters = []
max_attempts = 12

while True:
    hidden_word = ''.join([letter if letter in guessed_letters else '_' for letter in word])
    print(f"Word: {hidden_word}")

    if '_' not in hidden_word:
        print("Congratulations! You guessed the word correctly!")
        print("The word was", word, ":)")
        print("Well, that was fun! Let's move on to the next part of the script!")
        break

    if len(guessed_letters) == max_attempts:
        print("Game Over! You ran out of attempts.")
        print("Just so you know, the final word was", word, ":)")
        print("Well, that was fun! Let's move on to the next part of the script!")
        break

    guess = input("Guess a letter: ")
    if guess in guessed_letters:
        print("You already guessed that letter. Try again.")
        continue

    guessed_letters.append(guess)

    if guess not in word:
        print("Wrong guess!")
        print(f"Attempts remaining: {max_attempts - len(guessed_letters)}")

# chatgpt
print(" ")
print("You know that AI that people are using to cheat, ChatGPT?")
print("Well, because I'm a NICE script, I'll allow you to ask ChatGPT 10 queries before I have to move on :)")

import openai

openai.api_key = '<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>'

# query 1
def get_completion(prompt, model="davinci"):
    response = openai.Completion.create(
        engine=model,
        prompt=prompt,
        temperature=0.7,
        max_tokens=100,
        n=1,
        stop=None
    )
    
    return response.choices[0].text.strip()

prompt = input("What would you like to ask ChatGPT? ")
response = get_completion(prompt)
print(response)

# query 2
import openai

openai.api_key = '<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>'

def get_completion(prompt, model="davinci"):
    response = openai.Completion.create(
        engine=model,
        prompt=prompt,
        temperature=0.7,
        max_tokens=100,
        n=1,
        stop=None
    )
    
    return response.choices[0].text.strip()

prompt = input("What would you like to ask ChatGPT? ")
response = get_completion(prompt)
print(response)

# query 3
import openai

openai.api_key = '<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>'

def get_completion(prompt, model="davinci"):
    response = openai.Completion.create(
        engine=model,
        prompt=prompt,
        temperature=0.7,
        max_tokens=100,
        n=1,
        stop=None
    )
    
    return response.choices[0].text.strip()

prompt = input("What would you like to ask ChatGPT? ")
response = get_completion(prompt)
print(response)

# query 4
import openai

openai.api_key = '<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>'

def get_completion(prompt, model="davinci"):
    response = openai.Completion.create(
        engine=model,
        prompt=prompt,
        temperature=0.7,
        max_tokens=100,
        n=1,
        stop=None
    )
    
    return response.choices[0].text.strip()

prompt = input("What would you like to ask ChatGPT? ")
response = get_completion(prompt)
print(response)

# query 5
import openai

openai.api_key = '<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>'

def get_completion(prompt, model="davinci"):
    response = openai.Completion.create(
        engine=model,
        prompt=prompt,
        temperature=0.7,
        max_tokens=100,
        n=1,
        stop=None
    )
    
    return response.choices[0].text.strip()

prompt = input("What would you like to ask ChatGPT? ")
response = get_completion(prompt)
print(response)

# query 6
import openai

openai.api_key = '<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>'

def get_completion(prompt, model="davinci"):
    response = openai.Completion.create(
        engine=model,
        prompt=prompt,
        temperature=0.7,
        max_tokens=100,
        n=1,
        stop=None
    )
    
    return response.choices[0].text.strip()

prompt = input("What would you like to ask ChatGPT? ")
response = get_completion(prompt)
print(response)

# query 7
import openai

openai.api_key = '<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>'

def get_completion(prompt, model="davinci"):
    response = openai.Completion.create(
        engine=model,
        prompt=prompt,
        temperature=0.7,
        max_tokens=100,
        n=1,
        stop=None
    )
    
    return response.choices[0].text.strip()

prompt = input("What would you like to ask ChatGPT? ")
response = get_completion(prompt)
print(response)

# query 8
import openai

openai.api_key = '<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>'

def get_completion(prompt, model="davinci"):
    response = openai.Completion.create(
        engine=model,
        prompt=prompt,
        temperature=0.7,
        max_tokens=100,
        n=1,
        stop=None
    )
    
    return response.choices[0].text.strip()

prompt = input("What would you like to ask ChatGPT? ")
response = get_completion(prompt)
print(response)

# query 9
import openai

openai.api_key = '<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>'

def get_completion(prompt, model="davinci"):
    response = openai.Completion.create(
        engine=model,
        prompt=prompt,
        temperature=0.7,
        max_tokens=100,
        n=1,
        stop=None
    )
    
    return response.choices[0].text.strip()

prompt = input("What would you like to ask ChatGPT? ")
response = get_completion(prompt)
print(response)

# query 10
import openai

openai.api_key = '<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>'

def get_completion(prompt, model="davinci"):
    response = openai.Completion.create(
        engine=model,
        prompt=prompt,
        temperature=0.7,
        max_tokens=100,
        n=1,
        stop=None
    )
    
    return response.choices[0].text.strip()

prompt = input("What would you like to ask ChatGPT? ")
response = get_completion(prompt)
print(response)
print(" ")
print("Alright, OpenAI, your time in the spotlight of this script is over. Let's move on!")