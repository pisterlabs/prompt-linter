import pygame
import numpy as np
from dataclasses import dataclass
import os
import math
import pickle

import hashlib

from UI.piece_draw_screen import TOTAL_EXPECTED_PIECES, WALK, BLACK, WHITE
from UI.board_create_screen import PIECE_HEIGHT, PIECE_WIDTH, BLACK_PIECE, WHITE_PIECE
from langchain.chat_models import ChatOpenAI
from Helpers.prompts import get_next_move_prompt
import asyncio

# Initializing the llm model with the openai api key stored in the .env file
llm = ChatOpenAI(openai_api_key=os.getenv("OPENAI_API_KEY"))


class GameLogic:
    def __init__(self, rows=8, cols=8):
        self.piece_alignment = np.zeros((rows, cols))
        self.piece_position = np.zeros((rows, cols))
        self.initial_piece_position = np.zeros((rows, cols))
        self.initial_piece_alignment = np.zeros((rows, cols))
        self.pieces = {}
        self.selected_piece = None
        self.piece_can_move_to = set()
        self.turn = WHITE_PIECE
        self.game_history = []
        self.kings = []
        self.game_over = False
        self.winner = None
        self.name = None

    def ai_move(self):
        """Gets the next move from the ai and handles it by calling the handle_press function with the correct parameters"""
        possible_moves = self.get_all_possible_moves()
        if os.path.exists(os.path.join(os.getcwd(), "Histories", f"{self.name}.txt")):
            with open(
                os.path.join(os.getcwd(), "Histories", f"{self.name}.txt"), "r"
            ) as f:
                past_games = f.read()
        else:
            past_games = ""
        prompt = get_next_move_prompt(
            init_piece_position=self.initial_piece_position,
            init_piece_alignment=self.initial_piece_alignment,
            game_history=self.game_history,
            possible_moves=possible_moves,
            past_games=past_games,
        )

        try:
            next_move = llm.predict(prompt)

            parsed_move = next_move.split("->")

            starting = parsed_move[0].strip().split("=")[1].strip()
            row1, col1 = starting.strip("")[1], starting.strip("")[-2]
            ending = parsed_move[1]
            row2, col2 = ending.strip("")[1], ending.strip("")[-2]
        except:
            next_move = possible_moves[0]
            parsed_move = next_move.split("->")

            starting = parsed_move[0].strip().split("=")[1].strip()
            row1, col1 = starting.strip("")[1], starting.strip("")[-2]
            ending = parsed_move[1]
            row2, col2 = ending.strip("")[1], ending.strip("")[-2]

        self.handle_press(int(row1), int(col1))
        self.handle_press(int(row2), int(col2))

    def handle_press(self, row, col):
        """Handles the press of a piece on the board

        If there is no piece selected, then it selects the piece

        If there is a piece selected, then it checks if the piece can move to the selected position

        If it can, then it moves the piece to the selected position"""

        mesh_size = self.piece_position.shape[0]
        if 0 <= row < mesh_size and 0 <= col < mesh_size:
            print(f"piece: {self.pieces.get((row, col), None)}")
            if self.selected_piece is not None:
                print(f"Currently the piece is at {self.selected_piece.position}")
                if (row, col) in self.piece_can_move_to:
                    self.game_history.append(
                        f'{"w" if self.turn == WHITE_PIECE else "b"}{self.selected_piece.hash}={self.selected_piece.position}->{row, col}'
                    )

                    del self.pieces[
                        (
                            self.selected_piece.position[0],
                            self.selected_piece.position[1],
                        )
                    ]
                    if self.pieces.get((row, col), None) is not None:
                        if self.pieces[(row, col)].is_king:
                            self.game_over = True
                            self.winner = (
                                "White"
                                if self.selected_piece.color == WHITE_PIECE
                                else "Black"
                            )
                            self.game_history.append(f"\n{self.winner} wins\n")
                    self.piece_position[self.selected_piece.position] = 0
                    self.piece_position[row, col] = self.selected_piece.hash
                    self.piece_alignment[self.selected_piece.position] = 0
                    self.piece_alignment[row, col] = self.selected_piece.color
                    self.selected_piece.position = (row, col)
                    print(self.selected_piece.position)
                    self.piece_can_move_to = []
                    self.pieces[(row, col)] = self.selected_piece
                    self.selected_piece = None
                    self.turn = WHITE_PIECE if self.turn == BLACK_PIECE else BLACK_PIECE
                else:
                    self.selected_piece = None
                    self.piece_can_move_to = []
            else:
                self.selected_piece = self.pieces.get((row, col), None)
                print(f"selected piece: {self.selected_piece}")
                if self.selected_piece:
                    if self.selected_piece.color != self.turn:
                        self.selected_piece = None
                        return
                    self.piece_can_move_to = self.selected_piece.get_valid_moves(
                        self.piece_position, (row, col), self.piece_alignment
                    )
                    print(f"piece can move to: {self.piece_can_move_to}")

    def get_all_possible_moves(self):
        """Returns all possible moves for the current player"""
        moves = []
        for piece in self.pieces.values():
            if piece.color == self.turn:
                valid_moves = piece.get_valid_moves(
                    self.piece_position, piece.position, self.piece_alignment
                )
                piece_valid_moves = [
                    f'{"w" if piece.color == WHITE_PIECE else "b"}{piece.hash}={piece.position}->{move}'
                    for move in valid_moves
                ]
                moves.extend(piece_valid_moves)
        return moves

    def get_pieces_from_hash(self, piece_dictionary):
        """Gets the pieces from the hashes stored in the pickle file"""
        self.pieces = {}

        with open(os.path.join(os.getcwd(), "pieces.pkl"), "rb") as f:
            hash_to_piece = pickle.load(f)

            pieces = np.where(self.piece_position != 0)

            for i, piece in enumerate(zip(pieces[0], pieces[1])):
                r, c = piece
                self.pieces[(r, c)] = piece_dictionary[
                    hash_to_piece[self.piece_position[r][c]]
                ].copy()
                self.pieces[(r, c)].position = (r, c)
                self.pieces[(r, c)].color = self.piece_alignment[r][c]
                # When a piece is black, we rotate the movement matrix by 180 degrees to account for the fact that the board is flipped
                if self.pieces[(r, c)].color == BLACK_PIECE:
                    self.pieces[(r, c)].movement = np.rot90(
                        self.pieces[(r, c)].movement, 2
                    )

        # Turning the king pieces into kings

        for king_position in self.kings:
            self.pieces[tuple(king_position)].is_king = True
