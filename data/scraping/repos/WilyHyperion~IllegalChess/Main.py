from flask import app, current_app, Flask, request, send_from_directory
import os
import flask
import json 
import random
import re
#import Scrapper
import GPTScrapper
#import cohere
promptStart = f"""Let's play a game of Chess. I will be white you will be black. I will give you a list of moves that have already happend in normal chess notation. You will respond with a single move, in standard chess notation. You will not include any other comments, just the single move. If you think an move is illegal or invaild, still provide a move like before Moves:"""
#promptStart = f""""Lets play chess. Moves so far: """
InUseId = []
GameMoves = {

}


app = Flask("Chess")
@app.route("/favicon.ico")
def imag():
  return send_from_directory("Pages/img/anarchy.png", "anarchy.png")
@app.route("/games/img/<path:path>")
def img(path):
    print(path) 
    return send_from_directory("Pages/img", path)
@app.route("/")
def index():
    return open("Pages/index.html").read()
@app.route("/game")
def game():
    return open("Pages/game.html").read()
@app.route("/games/<id>/move", methods = ['POST'])
def move(id):
    id = int(id)
    if id not in InUseId:
        return open("Pages/invalid.html").read()
    GameMoves[id].append(request.json['move'])
    return getNextMove(id)
@app.route("/games/<id>", methods = ['GET'])
def games(id):
    id = int(id)
    if id not in InUseId:
        return open("Pages/invalid.html").read()
    return open("Pages/game.html").read()
@app.route("/games/<id>", methods = ['POST'])
def postGame(id):
    numid = int(id)
    GameMoves[numid].append(request.json['move'])
    n = getNextMove(id)
    GameMoves[numid].append(n)
    return json.dumps({"move": n}) 
@app.route("/newgame", methods=['POST'] )
def newgame():
    id = getNewGameId()
    print(InUseId)
    GameMoves[id] = []
    return json.dumps({"id": id} )
@app.route("/games/<id>/end", methods = ['GET'])
def endgame(id):
    id = int(id)
    if id not in InUseId:
        return open("Pages/invalid.html").read()
    InUseId.remove(id)
    del GameMoves[id]
    return "Game Ended"
def getNewGameId():
    id = random.randint(100, 999)
    while id in InUseId:
        id = random.randint(100, 999)
    InUseId.append(id)
    return id
def getNextMove(id, special = False):
    id = int(id)
    moves = GameMoves[id]
    p = promptStart;
    for i in moves:
        p += i + "  "
    print(p)
    s = GPTScrapper.gettext(p)
    print(s)
    # split s by spaces and find the first one that is a valid move. find a valid move by seeing if there is a number and a letter in the the text anywhere
    if " " in s:
      while " " in s:
        s = GPTScrapper.gettext(p + ".  Please only provide a single move in standard chess notation and nothing else.")
        print(s + "[new]")
      """ move = s.split(" ")
      hasInt = False;
      hasStr = False;
      for i in move:
        for j in i:
          if j.isnumeric():
            hasInt = True
          if j.isalpha():
            hasStr = True
        if(hasInt and hasStr):
          s = move.index(i)
        else:
          hasInt = False
          hasStr = False
      if(s == temp):
        s = "e4" """
    
    moves.append(s)
    return s
@app.route("/games/<id>/comments", methods = ['GET'])
def getComments(id):
    id = int(id)
    if id not in InUseId:
        return "bad game id"
    return GPTScrapper.getUnRealtedText("Provide a comment on the chess game, with these moves. Make your comments helpful for someone trying to learn chess while also being less than 30 words: " + str(GameMoves[id]))

