import time
import threading
import pyttsx3
from langchain.tools import tool, BaseTool
from typing import Type
from pydantic import BaseModel, Field
import chess
import chess.engine
import json
from datetime import datetime


"""
Chess tool

BOARD is a constant containing the board state

tools:
- play a human move, then (stockfish) play a move
- get the board state
- reset the board

"""

board_path = "data/chess_info.json"


def update_board(board):
    with open(board_path, "w") as file:
        json.dump(
            {
                "fen": board.fen(),
                "last_move_date": datetime.now().strftime(r"%Y-%m-%d %H:%M"),
            },
            file,
        )


def get_chess_info():
    with open(board_path, "r") as file:
        data = json.load(file)
        fen_string = data["fen"]
        date_string = data["last_move_date"]
    return fen_string, date_string


def get_board():
    fen_string, date_string = get_chess_info()
    BOARD = chess.Board(fen=fen_string)
    return BOARD


# fen_string, date_string = get_chess_info()
# BOARD = chess.Board(fen=fen_string)
# LAST_MOVE = date_string
STOCKFISH = chess.engine.SimpleEngine.popen_uci(r"C:\Users\JeanLELONG\chess_engines\stockfish_15.1\stockfish.exe")


class MoveInput(BaseModel):
    """Inputs for PlayMove"""

    move: str = Field(description="coup d'échecs en notation algébrique (e4, Nf5, ...)")


class PlayMove(BaseTool):
    name = "PlayMove"
    description = "joue le coup donné sur l'échiquier, puis joue le coup de stockfish"
    args_schema: Type[BaseModel] = MoveInput

    def _run(self, move: str) -> str:
        BOARD = get_board()

        try:
            human_move = BOARD.parse_san(move)
        except chess.IllegalMoveError:
            return f"{move} is not a legal move in the current position."

        algebraic_human_move = BOARD.san(human_move)
        BOARD.push(human_move)
        engine_move = STOCKFISH.play(BOARD, chess.engine.Limit(time=0.1)).move
        algebraic_engine_move = BOARD.san(engine_move)
        BOARD.push(engine_move)

        update_board(BOARD)

        return f"Jean played {algebraic_human_move} and Cyrano played {algebraic_engine_move}."

    def _arun(self, move: str):
        raise NotImplementedError(f"{self.name} does not support async")


class GetBoardState(BaseTool):
    name = "GetBoardState"
    description = "renvoie le plateau d'échec sous forme de FEN, ainsi que la date du dernier coup joué"

    def _run(self, *args, **kwags) -> str:
        fen_string, date_string = get_chess_info()
        return f"fen = {fen_string}, dernier coup joué le {date_string}"

    def _arun(self):
        raise NotImplementedError(f"{self.name} does not support async")


class ResetBoard(BaseTool):
    name = "ResetBoard"
    description = "utile pour réinitialiser le plateau d'échec"

    def _run(self, *args, **kwags) -> str:
        BOARD = chess.Board()
        update_board(BOARD)
        fen_string, date_string = get_chess_info()
        return f"les pièces sont à leur position de départ : {BOARD.fen()}"

    def _arun(self):
        raise NotImplementedError(f"{self.name} does not support async")
