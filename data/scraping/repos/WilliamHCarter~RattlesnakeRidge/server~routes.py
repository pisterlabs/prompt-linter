import uuid
from flask import Flask, request, jsonify
from server.game import play_game, initialize_game  
from server import game_states, app, logger
from server.commands import marshal_command

import logging

logger = logging.getLogger(__name__)


@app.route('/start', methods=['GET'])
def start_game():
    # Generating a unique game ID
    game_id = str(uuid.uuid4())
        
    # Create the llm to use for this game
    # todo:: get the user's provided API key ;)
    from langchain.chat_models import FakeListChatModel
    llm = FakeListChatModel(
        verbose=True,
        responses=[
            "Hi there, I'm talking to you.",
            "This is a response",
            "I say something else too!",
            "Ok, goodbye now!",
        ],
    )
    
    # Initializing the game state
    game_states[game_id] = initialize_game(llm=llm)

    logger.info("Created a new game with id %s", game_id)
    return jsonify(game_id=game_id, message="Game started!")

@app.route('/play/<game_id>', methods=['POST'])
def play(game_id):
    user_input = request.json.get('input')
    game_state = game_states.get(game_id)
    
    if game_state is None:
        logger.warning("`/play` called with an invalid game id %s", game_id)
        return jsonify(error="Invalid game ID"), 400

    if not game_state.is_input_valid(user_input):
        logger.warning("invalid user input provided \"%s\" for game id %s", user_input, game_id)
        return jsonify(error="Bad user input"), 400
    
    command = play_game(game_state, user_input)
    return jsonify(response=marshal_command(command))

@app.route('/end/<game_id>', methods=['POST'])
def end_game(game_id):
    if game_id in game_states:
        logger.info("game with id %s ended", game_id)
        del game_states[game_id]
        return jsonify(message="Game data cleared!")
    else:
        logger.info("attempted to end game with id %s, but it does not exist", game_id)
        return jsonify(error="Invalid game ID"), 400

@app.route('/test', methods=['GET'])
def test():
    return jsonify({"message": "Success"})


if __name__ == '__main__':
    app.run(debug=True)
