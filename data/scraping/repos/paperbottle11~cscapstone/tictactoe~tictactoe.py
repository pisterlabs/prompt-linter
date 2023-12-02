import openai
from pydantic import BaseModel
import json

# Load API Key
with open("../config.txt", "r") as f:
    api_key = f.read().strip()
    openai.api_key = api_key

# GPT Function Structure
class TicTacToeResponse(BaseModel):
    row: int
    column: int

# Generate Board
board = [[" " for i in range(3)] for i in range(3)]

# Generates move using GPT-4
def generateMove(board):
    apiResponse = openai.ChatCompletion.create(
        model="gpt-4-0613",
        messages=[
            {"role": "system", "content": "You are an aid for playing tic tac toe. Pay attention to the user's instructions and respond accordingly."},
            {"role": "user", "content": """I will give you a tic tac toe board in the form of a 2D python array.  Each inner list represents a row, and each element in the inner list is a column in that row.  You need to tell me the best move to make.  You are "O". Only place your O on an empty blank space. Only respond with numbers between 0 and 2.  The board is: """ + str(board)},
        ],
        functions=[
            {
            "name": "generate_move",
            "description": "Return the row and column between 0 and 2 for the best move to make that does not go over a taken space",
            "parameters": TicTacToeResponse.model_json_schema()
            }
        ],
        function_call={"name": "generate_move"}
    )

    output = apiResponse.choices[0]["message"]["function_call"]["arguments"]
    move = json.loads(output)
    return move

# Check if the game is over and return the winner
def checkWin(board):
    for i in range(3):
        if board[i][0] == board[i][1] == board[i][2] and board[i][0] != " ":
            return board[i][0]
        elif board[0][i] == board[1][i] == board[2][i] and board[0][i] != " ":
            return board[0][i]
    if board[0][0] == board[1][1] == board[2][2] and board[0][0] != " ":
        return board[0][0]
    elif board[0][2] == board[1][1] == board[2][0] and board[0][2] != " ":
        return board[0][2]
    return " "

def checkTie(board):
    for i in range(len(board)):
        if " " in board[i]:
            return False
    return True

def printBoard(board):
    for i in range(len(board)):
        print(board[i])

def place(board, row, col, player):
    if board[int(row)][int(col)] == " ":
        board[int(row)][int(col)] = player
        return True
    else:
        return False

# Game Loop
while not checkTie(board) and checkWin(board) == " ":
    printBoard(board)
    player = input("Enter row and col (1-3): ")
    row = int(player[0]) - 1
    col = int(player[1]) - 1
    
    while row > 2 or col > 2 or row < 0 or col < 0 or board[row][col] != " " or len(player) != 2:
        player = input("Enter row and col (1-3): ")
        row = int(player[0]) - 1
        col = int(player[1]) - 1
    
    place(board, row, col, "X")

    if checkTie(board) == True or checkWin(board) != " ": break
    printBoard(board)

    move = generateMove(board)
    print("Opp Move: " + str(move))
    placed = place(board, move["row"], move["column"], "O")

    # If the generated move is invalid, keep generating moves until a valid one is created
    while not placed:
        move = generateMove(board)
        print("Opp Move: " + str(move))
        placed = place(board, move["row"], move["column"], "O")

printBoard(board)
print("Winner: " + checkWin(board))