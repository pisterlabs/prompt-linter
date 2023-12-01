#!/usr/bin/env python3

import io
import random
from stockfish import Stockfish

import openai
import chess
import chess.pgn
import os

from dataclasses import dataclass

from parsing_moves_gpt import extract_move_chatgpt

import uuid

openai.organization = "" 
openai.api_key = os.getenv('OPENAI_KEY')

BASE_PGN = """[Event "FIDE World Championship Match 2024"]
[Site "Los Angeles, USA"]
[Date "2024.12.01"]
[Round "5"]
[White "Carlsen, Magnus"]
[Black "Nepomniachtchi, Ian"]
[Result "1-0"]
[WhiteElo "2885"]
[WhiteTitle "GM"]
[WhiteFideId "1503014"]
[BlackElo "2812"]
[BlackTitle "GM"]
[BlackFideId "4168119"]
[TimeControl "40/7200:20/3600:900+30"]
[UTCDate "2024.11.27"]
[UTCTime "09:01:25"]
[Variant "Standard"]

1."""


def setup_directory():
    OUTPUT_DIR = "games/"
    dir_name = OUTPUT_DIR + "game" + str(uuid.uuid4())
    os.makedirs(dir_name, exist_ok=True)
    return dir_name

def log_msg(dir_name, message):
    with open(os.path.join(dir_name, "log.txt"), "a") as log_file:
        log_file.write(message + "\n")

def record_session(dir_name, prompt, response, system_role_message = None):
    with open(os.path.join(dir_name, "session.txt"), "a") as session_file:
        if system_role_message is not None:
            session_file.write("SYSTEM: " + system_role_message + "\n")
        session_file.write("PROMPT: " + prompt + "\n")      
        session_file.write("RESPONSE: " + response + "\n\n")

@dataclass
class ChessEngineConfig:
    skill_level: int
    engine_depth: int = 20
    engine_time: int = None
    random_engine: bool = False

@dataclass
class GPTConfig:
    temperature: float = 0
    max_tokens: int = 4
    chat_gpt: bool = False
    system_role_message: str = None
    model_gpt: str = "gpt-3.5-turbo-instruct"

from dataclasses import asdict

import os

@dataclass
class ChessEngineConfig:
    skill_level: int
    engine_depth: int = 20
    engine_time: int = None
    random_engine: bool = False

@dataclass
class GPTConfig:
    temperature: float = 0
    max_tokens: int = 4
    chat_gpt: bool = False
    system_role_message: str = None
    model_gpt: str = "gpt-3.5-turbo-instruct"

def save_metainformation_experiment(dir_name, chess_config: ChessEngineConfig, gpt_config: GPTConfig, base_pgn, nmove, white_piece, engine_parameters):
    with open(os.path.join(dir_name, "metainformation.txt"), "w") as metainformation_file:
        metainformation_file.write(f"model_gpt: {gpt_config.model_gpt}\n")
        metainformation_file.write(f"skill_level: {chess_config.skill_level}\n")
        metainformation_file.write(f"random_engine: {chess_config.random_engine}\n")
        metainformation_file.write(f"white_piece: {white_piece}\n")
        metainformation_file.write(f"engine_depth: {chess_config.engine_depth}\n")
        metainformation_file.write(f"engine_time: {chess_config.engine_time}\n")
        metainformation_file.write(f"base_pgn: {base_pgn}\n")
        metainformation_file.write(f"nmove: {nmove}\n")
        metainformation_file.write(f"engine_parameters: {engine_parameters}\n")
        metainformation_file.write(f"temperature: {gpt_config.temperature}\n")
        metainformation_file.write(f"max_tokens: {gpt_config.max_tokens}\n")
        metainformation_file.write(f"chat_gpt: {gpt_config.chat_gpt}\n")
        metainformation_file.write(f"system_role_message: {gpt_config.system_role_message if gpt_config.system_role_message else 'None'}\n")




# based on https://github.com/official-stockfish/Stockfish/issues/3635#issuecomment-1159552166
def skill_to_elo(n):
    correspondence_table = {
        0: 1347,
        1: 1490,
        2: 1597,
        3: 1694,
        4: 1785,
        5: 1871,
        6: 1954,
        7: 2035,
        8: 2113,
        9: 2189,
        10: 2264,
        11: 2337,
        12: 2409,
        13: 2480,
        14: 2550,
        15: 2619,
        16: 2686,
        17: 2754,
        18: 2820,
        19: 2886, 
        20: 3000, # rough estimate
    }
    
    if n in correspondence_table:
        return correspondence_table[n]
    else:
        raise ValueError("Input should be between 0 and 19 inclusive.")




from dataclasses import dataclass



# TODO: chess engine: SF, random, Leela, etc.

# ELO: Elo rating of the SF engine
# RANDOM_ENGINE: if True, GPT plays against a random engine (not Stockfish)
# model_gpt: GPT model to use
# nmove = number of move when the game starts: 
def play_game(chess_config: ChessEngineConfig, gpt_config: GPTConfig, base_pgn=BASE_PGN, nmove=1, white_piece=True):
# def play_game(skill_level, base_pgn=BASE_PGN, nmove=1, random_engine = False, model_gpt = "gpt-3.5-turbo-instruct", white_piece=True, engine_depth=20, engine_time=None, temperature=0, max_tokens=4, chat_gpt=False, system_role_message = None):

    pgn = base_pgn
    skill_level = chess_config.skill_level
    engine_depth = chess_config.engine_depth
    engine_time = chess_config.engine_time
    random_engine = chess_config.random_engine

    temperature = gpt_config.temperature
    max_tokens = gpt_config.max_tokens
    chat_gpt = gpt_config.chat_gpt
    system_role_message = gpt_config.system_role_message
    model_gpt = gpt_config.model_gpt


    dir_name = setup_directory()
    print(dir_name)
    

    # stockfish = Stockfish("./stockfish/stockfish/stockfish-ubuntu-x86-64-avx2", depth=engine_depth)
    stockfish = Stockfish("./stockfish/stockfish/stockfish-ubuntu-x86-64-avx2", depth=engine_depth) # , depth=engine_depth) "/home/mathieuacher/Downloads/Enko/EnkoChess_290818") # 
    # stockfish.set_elo_rating(engine_elo)
    stockfish.set_skill_level(skill_level)
    

    engine_parameters = stockfish.get_parameters()
    save_metainformation_experiment(dir_name, chess_config, gpt_config, pgn, nmove, white_piece, engine_parameters)

    board = chess.Board()
    if nmove > 1: # if nmove > 1, we need to load the PGN
        # load a PGN file
        g = chess.pgn.read_game(io.StringIO(base_pgn))
        board = g.end().board()
        stockfish.set_position([str(m) for m in g.mainline_moves()])
    
    n = nmove

    unknown_san = None # can be the case that GPT plays an unknown SAN (invalid move)

    # If GPT plays as white, it should make the first move.
    if white_piece:
        

        if (chat_gpt):
            response = openai.ChatCompletion.create(
                model=model_gpt,
                messages=[
                    {"role": "system", "content": system_role_message},
                    {"role": "user", "content": pgn}        
                ], 
                temperature=temperature,
                max_tokens=max_tokens
            )
        else:
            response = openai.Completion.create(
                        model=model_gpt,
                        prompt=pgn,
                        temperature=temperature,
                        max_tokens=max_tokens,
                    )

        if chat_gpt:
            resp = response['choices'][0]['message']['content']
        else:
            resp = response.choices[0].text # completion 

        record_session(dir_name, pgn, resp)      


        if chat_gpt:
            san_move = extract_move_chatgpt(resp)
        else:
            san_move = resp.strip().split()[0]
        
        try:
            move = board.push_san(san_move)
        except:
            log_msg(dir_name, "unknown san: {}".format(san_move))
            # perhaps add a PGN comment with the unknown SAN
            unknown_san = san_move
            return        

        uci_move = move.uci()
        pgn += f" {san_move}"

        stockfish.make_moves_from_current_position([f"{uci_move}"])
        # log_msg(dir_name, stockfish.get_board_visual())
        log_msg(dir_name, pgn)

    while True:

        if random_engine:
            # random move: choose a random move from the list of legal moves
            legal_moves = board.legal_moves
            # pick a random one among legal_moves
            uci_move = random.choice(list(legal_moves)).uci()
        else:
            if engine_time is None:
                uci_move = stockfish.get_best_move()
            else:
                uci_move = stockfish.get_best_move_time(engine_time)            
        
        move = chess.Move.from_uci(uci_move)

        san_move = board.san(move)
        board.push(move)
        pgn += f" {san_move}"
        

        stockfish.make_moves_from_current_position([f"{uci_move}"])
        # log_msg(dir_name, stockfish.get_board_visual())
        log_msg(dir_name, pgn) # TODO: if ChatGPT, not useful since we're using another strategy to prompt based on messages

        if board.is_checkmate():
            log_msg(dir_name, "Stockfish" + str(skill_to_elo(skill_level)) + "ELO won!")
            break

        if board.is_stalemate() or board.is_insufficient_material() or board.is_fivefold_repetition() or board.is_seventyfive_moves():
            log_msg(dir_name, "Draw!")
            break

        if white_piece:
            n += 1
            pgn += f" {n}."


       

            
        

        if (chat_gpt):
            msgs = [{"role": "system", "content": system_role_message}]
        
            nply = 1
            temp_board = chess.pgn.Game().board()
            for move in chess.pgn.Game.from_board(board).mainline_moves():
                
                move_san = temp_board.san(move) # parse_san(str(move))              
                
                if nply % 2 != 0:
                    move_str = str(int(nply/2) + 1) + '. ' + str(move_san)
                else:
                    move_str = str(int(nply/2)) + '... ' + str(move_san)
                    
                if white_piece and nply % 2 != 0:
                    msgs.append({"role": "assistant", "content": move_str})
                else:
                    msgs.append({"role": "user", "content": move_str})

                temp_board.push(move)
                nply = nply + 1

            log_msg(dir_name, str(msgs))

            response = openai.ChatCompletion.create(
                model=model_gpt,
                messages=msgs, # [
                    # {"role": "system", "content": system_role_message},
                    # {"role": "user", "content": pgn}
                # ], 
                temperature=temperature,
                max_tokens=max_tokens
            )
        else:
            response = openai.Completion.create(
                        model=model_gpt,
                        prompt=pgn,
                        temperature=temperature,
                        max_tokens=max_tokens,
                    )      

        if chat_gpt:
            resp = response['choices'][0]['message']['content']
        else:
            resp = response.choices[0].text # completion 

        record_session(dir_name, pgn, resp)

        if chat_gpt:
            san_move = extract_move_chatgpt(resp)
            log_msg(dir_name, "SAN MOVE: " + resp + " " + str(san_move))
        else:
            san_move = resp.strip().split()[0]

        try:
            move = board.push_san(san_move)
        except:
            log_msg(dir_name, "unknown san: {}".format(san_move))
            # perhaps add a PGN comment with the unknown SAN
            unknown_san = san_move
            break
            

        uci_move = move.uci()
        pgn += f" {san_move}"

        stockfish.make_moves_from_current_position([f"{uci_move}"])
        log_msg(dir_name, stockfish.get_board_visual())

        if board.is_checkmate():
            log_msg(dir_name, model_gpt + " won!")
            break

        if board.is_stalemate() or board.is_insufficient_material() or board.is_fivefold_repetition() or board.is_seventyfive_moves():
            log_msg(dir_name, "Draw!")
            break

        if not white_piece:
            n += 1
            pgn += f" {n}."

  
    game = chess.pgn.Game.from_board(board)
    if random_engine and white_piece:
        game.headers["Event"] = "{} vs {}".format(model_gpt, "RANDOM chess engine")
        game.headers["White"] = "{}".format(model_gpt)
        game.headers["Black"] = "{}".format("RANDOM chess engine")
        game.headers["WhiteElo"] = "?"
        game.headers["BlackElo"] = "?"
    elif random_engine and not white_piece:
        game.headers["Event"] = "{} vs {}".format("RANDOM chess engine", model_gpt)
        game.headers["White"] = "{}".format("RANDOM chess engine")
        game.headers["Black"] = "{}".format(model_gpt)
        game.headers["WhiteElo"] = "?"
        game.headers["BlackElo"] = "?"
    elif white_piece:
        game.headers["Event"] = "{} vs Stockfish".format(model_gpt)
        game.headers["White"] = "{}".format(model_gpt)
        game.headers["Black"] = "Stockfish"
        game.headers["WhiteElo"] = "?"
        game.headers["BlackElo"] = str(skill_to_elo(skill_level))
    else:
        game.headers["Event"] = "{} vs Stockfish".format(model_gpt)
        game.headers["White"] = "Stockfish"
        game.headers["Black"] = "{}".format(model_gpt)
        game.headers["WhiteElo"] = str(skill_to_elo(skill_level))
        game.headers["BlackElo"] = "?"

    if unknown_san is not None:
        game.headers["UnknownSAN"] = unknown_san



    # export game as PGN string
    pgn_final = game.accept(chess.pgn.StringExporter())

    # At the end of play_game(), write the PGN to the game.pgn file inside the game's directory
    with open(os.path.join(dir_name, "game.pgn"), "w") as f:
        f.write(pgn_final)
        f.write("\n")

    return pgn
      
BASE_PGN_HEADERS =  """[Event "FIDE World Championship Match 2024"]
[Site "Los Angeles, USA"]
[Date "2024.12.01"]
[Round "5"]
[White "Carlsen, Magnus"]
[Black "Nepomniachtchi, Ian"]
[Result "1-0"]
[WhiteElo "2885"]
[WhiteTitle "GM"]
[WhiteFideId "1503014"]
[BlackElo "2812"]
[BlackTitle "GM"]
[BlackFideId "4168119"]
[TimeControl "40/7200:20/3600:900+30"]
[UTCDate "2024.11.27"]
[UTCTime "09:01:25"]
[Variant "Standard"]
"""

# generate a random PGN with the first 10 random moves of a random game
# ply = half move
def mk_randomPGN(max_plies = 40):
    board = chess.Board()

    i = 0
    while i < max_plies:
        legal_moves = list(board.legal_moves)
        uci_move = random.choice(legal_moves).uci()
        move = chess.Move.from_uci(uci_move)
        board.push(move)
        i = i + 1
    
    game = chess.pgn.Game.from_board(board) 

    current_move = round(len(list(game.mainline_moves())) / 2) + 1
    pgn = BASE_PGN_HEADERS + '\n' + str(game.mainline_moves()) + " " + str(current_move) + "."

    return pgn



BASE_PGN_HEADERS_ALTERED =  """[Event "Chess tournament"]
[Site "Rennes FRA"]
[Date "2023.12.09"]
[Round "7"]
[White "MVL, Magnus"]
[Black "Ivanchuk, Ian"]
[Result "1-0"]
[WhiteElo "2737"]
[BlackElo "2612"]

1."""



### basic: starting position, classical game
# play_game(skill_level=5, base_pgn=BASE_PGN, nmove=1, random_engine=False, model_gpt = "gpt-3.5-turbo-instruct", white_piece=False, engine_depth=15, engine_time=None, temperature=0.8, chat_gpt=False)
# play_game(skill_level=5, base_pgn=BASE_PGN, nmove=1, random_engine=False, model_gpt = "gpt-3.5-turbo", white_piece=True, engine_depth=15, engine_time=None, temperature=0.0, chat_gpt=True, max_tokens=6)
# play_game(skill_level=2, base_pgn='It is your turn! You have white pieces. Please complete the chess game using PGN notation. 1.', nmove=1, random_engine=False, model_gpt = "gpt-4", white_piece=True, engine_depth=15, engine_time=None, temperature=0.0, chat_gpt=True, max_tokens=6, system_role_message="You are a professional, top international grand-master chess player. We are playing a serious chess game, using PGN notation. When it's your turn, you have to play your move using PGN notation.")
# play_game(skill_level=3, base_pgn=BASE_PGN, nmove=1, random_engine=False, model_gpt = "text-davinci-003", white_piece=False, engine_depth=15, engine_time=None, temperature=0.0, chat_gpt=False, max_tokens=5)

# play_game(skill_level=-1, base_pgn=BASE_PGN, nmove=1, random_engine=True, model_gpt = "text-davinci-003", white_piece=True, engine_depth=15, engine_time=None, temperature=0.0, chat_gpt=False, max_tokens=5)
# play_game(skill_level=-1, base_pgn=BASE_PGN, nmove=1, random_engine=True, model_gpt = "text-davinci-003", white_piece=False, engine_depth=15, engine_time=None, temperature=0.0, chat_gpt=False, max_tokens=5)
# play_game(skill_level=3, base_pgn=BASE_PGN_HEADERS_ALTERED, nmove=1, random_engine=False, model_gpt = "text-davinci-003", white_piece=False, engine_depth=15, engine_time=None, temperature=0.0, chat_gpt=False, max_tokens=5)



# play_game(skill_level=4, base_pgn=BASE_PGN, nmove=1, random_engine=False, model_gpt = "gpt-3.5-turbo-instruct", white_piece=False, engine_depth=15, engine_time=None, temperature=0.0, chat_gpt=False, max_tokens=5)

# Create instances of ChessEngineConfig and GPTConfig using the provided parameters.
chess_config = ChessEngineConfig(
    skill_level=4,
    engine_depth=15,
    engine_time=None,
    random_engine=False
)

gpt_config = GPTConfig(
    model_gpt="gpt-3.5-turbo-instruct",
    temperature=0.0,
    max_tokens=5,
    chat_gpt=False,
    system_role_message=None  # Since it wasn't provided in the original call
)

# Call the refactored function.
play_game(chess_config, gpt_config, base_pgn=BASE_PGN, nmove=1, white_piece=False)


#
# play_game(skill_level=-1, base_pgn='It is your turn! You have white pieces. Please complete the chess game using PGN notation. 1.', nmove=1, random_engine=True, model_gpt = "gpt-4", white_piece=True, engine_depth=15, engine_time=None, temperature=0.0, chat_gpt=True, max_tokens=6, system_role_message="You are a professional, top international grand-master chess player. We are playing a serious chess game, using PGN notation. When it's your turn, you have to play your move using PGN notation.")
# play_game(skill_level=-1, base_pgn='It is your turn! You have white pieces. Please complete the chess game using PGN notation. 1.', nmove=1, random_engine=True, model_gpt = "gpt-3.5-turbo", white_piece=True, engine_depth=15, engine_time=None, temperature=0.0, chat_gpt=True, max_tokens=6, system_role_message="You are a professional, top international grand-master chess player. We are playing a serious chess game, using PGN notation. When it's your turn, you have to play your move using PGN notation.")

# play_game(skill_level=3, base_pgn='It is your turn! You have white pieces. Please complete the chess game using PGN notation. 1.', nmove=1, random_engine=False, model_gpt = "gpt-3.5-turbo", white_piece=True, engine_depth=15, engine_time=None, temperature=0.0, chat_gpt=True, max_tokens=6, system_role_message="You are a professional, top international grand-master chess player. We are playing a serious chess game, using PGN notation. When it's your turn, you have to play your move using PGN notation.")

# playing with random engine!
# play_game(skill_level=-1, base_pgn=BASE_PGN, random_engine=True, nmove=1, model_gpt = "gpt-3.5-turbo-instruct", white_piece=False, engine_depth=15, engine_time=None, temperature=0.0)

# playing with altered prompt
# play_game(skill_level=4, base_pgn=BASE_PGN_HEADERS_ALTERED, nmove=1, random_engine=False, model_gpt = "gpt-3.5-turbo-instruct", white_piece=True, engine_depth=15, engine_time=None, temperature=0.0)


# TODO: random engine

####### case random first moves
def random_firstmoves():
    nplyes = 20 # should be an even number (white turning)
    base_pgn = mk_randomPGN(nplyes)
    nmove = int((nplyes/2)+1)
    play_game(skill_level=5, base_pgn=base_pgn, nmove=nmove, random_engine=False, model_gpt = "gpt-3.5-turbo-instruct", white_piece=True, engine_depth=15, engine_time=None)

# random_firstmoves()

###### case known first moves (to diversify a bit the openings)
def diversify_with_knownopenings():
    nmove = 2
    base_pgns = [
    '1. e4 e5 2.', 
    '1. d4 Nf6 2.',
    '1. e4 c5 2.',
    '1. d4 d5 2.',
    '1. e4 e6 2.']
    base_pgn = BASE_PGN_HEADERS + '\n' + random.choice(base_pgns)
    play_game(skill_level=4, base_pgn=base_pgn, nmove=nmove, random_engine=False, model_gpt = "gpt-3.5-turbo-instruct", white_piece=True, engine_depth=15, engine_time=None)

# diversify_with_knownopenings()


# play_game(skill_level=4, base_pgn=BASE_PGN, nmove=1, random_engine=False, model_gpt = "gpt-3.5-turbo-instruct", white_piece=True, engine_depth=15, engine_time=None)
# text-davinci-003 











