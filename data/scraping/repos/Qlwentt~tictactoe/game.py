import copy
from src.openai import OpenAI

from src.player import Player
from src.board import Board
from src.computer import Computer
from src.enums.game_modes import GameMode
from src.enums.moves import Moves
from src.enums.valid_positions import VALID_POSITIONS
from src.constants.constants import  GAME_MODES
from src.randomAI import RandomAI


class Game:
    def __init__(self, gameMode=None, player1=None, player2=None) -> None:
        self.gameMode = gameMode if gameMode else GameMode(Game.getInput("Choose your game mode\n1) 1 player\n2) 2 player\n", GAME_MODES))
        self.player1 = player1 or Player(Game.getInput("What is player 1's name? "), Moves.X.value)
        if not player2 and self.isTwoPlayerGame():
            self.player2 = Player(Game.getInput("What is player 2's name? "), Moves.O.value)
        else:
            self.player2 = player2 if player2 else Player('Computer', Moves.O.value)
        self.board = Board()
        self.isPlayer1Turn = True
        self.computer = Computer(self.player2, self.player1) if not self.isTwoPlayerGame() else None


    def say(message):
        print(message)

    def getInput(message,choices=None):
        variable = None
        if not choices:
            variable = input(message)
            return variable

        while not variable or variable not in choices:
            try:
                variable = int(input(message))
            except:
                variable = None
            if variable not in choices:
                print("Invalid choice")
            else:
                print(f'{choices[variable]} selected')

        return variable

    def isTwoPlayerGame(self):
        return self.gameMode == GameMode.TWO_PLAYER

    def drawBoard(self):
        self.board.draw()

    def drawValidMoves(self):
        self.board.drawMoves()

    def getValidMoves(self):
        return self.board.getValidMoves()

    def boardIsFull(self):
        return self.board.isFull()

    def updateTurn(self):
        self.isPlayer1Turn = not self.isPlayer1Turn

    def makeMove(self):
        if self.gameMode.value <= GameMode.TWO_PLAYER.value:
            player = self.player1 if self.isPlayer1Turn else self.player2

            if player == self.player2 and not self.isTwoPlayerGame():
                move = self.computer.makeMove(copy.deepcopy(self))
            else:
                move = Game.getInput(
                    f'{player.name}, make your move using the numbers that represent open spots: ',
                    { key: VALID_POSITIONS[key] for key in self.getValidMoves()} # dictionary of valid positions that aren't taken
                )
            self.board.markMove(VALID_POSITIONS[move],player.mark, player.tally)
            self.updateTurn()
        elif self.gameMode == GameMode.MINIMAX_VS_RANDOM:
            player = self.player1 if self.isPlayer1Turn else self.player2
            if player.name == 'Minimax':
                move = self.computer.makeMove(copy.deepcopy(self))
            else:
                randomAI = RandomAI()
                move = randomAI.makeMove(self)
            self.board.markMove(VALID_POSITIONS[move],player.mark, player.tally)
            self.updateTurn()
            return VALID_POSITIONS[move]
        elif self.gameMode == GameMode.OPENAI_VS_RANDOM:
            player = self.player1 if self.isPlayer1Turn else self.player2
            if player.name == 'OpenAI':
                openAI = OpenAI()
                move = openAI.makeMove(self)
                self.board.markMove(VALID_POSITIONS[move],player.mark, player.tally)
            else:
                randomAI = RandomAI()
                move = randomAI.makeMove(self)
                self.board.markMove(VALID_POSITIONS[move],player.mark, player.tally)
            self.updateTurn()


    def makeTheoreticalMove(self, move, player):
        self.board.markMove(VALID_POSITIONS[move], player.mark, player.tally)
        self.updateTurn()

    def undoMove(self,move, player):
        self.board.undoMove(move, player.tally)  
        self.updateTurn()

    def getWinner(self):
        winTally = self.board.getWinnerTallys()
        if not winTally:
            return None
        if winTally > 0:
            return self.player1
        else:
            return self.player2





