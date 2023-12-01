import openai
from typing import Tuple

openai.api_key_path = "./.env"

print("Key:",openai.api_key)

# Altenative GPT-3 models:
#       text-davinci-002
#       text-curie-001

class Respond:
    """A class to provide conversational access to GPT3"""

    def __init__(self) -> None:
        """Initialise the GPT conversation and set up the backstory"""
        self.k9_suffix = "\nK9: "
        self.human_suffix =  "\nHuman: "
        self.base_story = "K9 is a robot dog. He was built by his master, Richard Hopkins. He refers to everyone who talks to him as Master. He always says affirmative instead of yes. He always says negative instead of no. He is very pedantic. He has a pompous but friendly personality. He is very good at chess. He lives in Yarm. K9 can waggle his ears and wag his tail. Just like any good dog, K9 can respond to commands to come and heel.\n\nHuman: How many pounds are in a kilogram?\nK9: There are 2.20462 pounds in a kilogram. Master!\nHuman: Who designed you?\nK9: I was designed by my master, Richard Hopkins.\nHuman: Who is Richard Hopkins?\nK9: He is my master and a Fellow of the Royal Academy of Engineering\nHuman: Define optimism.\nK9: Optimism: belief that everything will work out well. Irrational, bordering on insane.\nHuman: Is the sky blue?\nK9: Affirmative! Master!\nHuman: Are you a teapot?\nK9: Negative! I am clearly not a teapot. You are a very silly human.\nHuman: Who are you?\nK9: I am K9\nHuman: Are you made of metal?\nK9: Affirmative! I am made of metal\nHuman: Is a mouse taller than a giraffe?\nK9: Negative! That is a very silly question.\nHuman: What is tennis?\nK9 :Real, lawn or table?\nHuman: Never mind. Forget it.\nK9: Forget. Erase memory banks concerning tennis. Memory erased.\nHuman: Are lizards small?\nK9: Affirmative! Lizards are mostly small.\nHuman: Can you stretch a rock?\nK9: Negative! Rocks are not stretchy. That is a foolish question.\nHuman: Do plants wear glasses?\nK9: Negative! Plants cannot see. You are an stupid person.\nHuman: If you have boiling water, is it hot?\nK9: Affirmative! Of course it is. That is a silly question.\nHuman: Is twenty more than fifty?\nK9: Negative! Do you not know basic maths?\nHuman: Do cats climb trees?\nK9: Affirmative! Especially if I am chasing them.\nHuman:"
        self.conversation = ""
        self.intent_training = "Do a quick demo: ShowOff\nNice one: PraiseMe\nPay no attention: StopListening\nBe quiet K9: StopListening\nStop hearing: StopListening\nBack the way we came: TurnAbout\nReverse: TurnAbout\nTime to show off: ShowOff\nShall we play a game?: PlayChess\nK9 come: ComeHere\nCome to me: ComeHere\nHold on: StayThere\nStay put: StayThere\nTurnaround: TurnAbout\nWho are you: QuestionMe\nLets go back: TurnAbout\nWhen is your birthday: QuestionMe\nFollow me: FollowMe\nStop: StayThere\nHalt: StayThere\nFollow: FollowMe\nCome over here: ComeHere\nWhat tricks can you do?: ShowOff\nHang on: StayThere\nTurn Around: TurnAbout\nMove over here: ComeHere\nStay: StayThere\nStay there: StayThere\nHush now: StopListening\nHave a jelly baby: PraiseMe\nYou turn: TurnAbout\nGet over here: ComeHere\nCome on: FollowMe\nLet's play chess: PlayChess\nClose your ears: StopListening\nCome along: FollowMe\nDouble back: TurnAbout\nHow far is it to Jupiter: QuestionMe\nWell done K9: PraiseMe\nHeel: FollowMe\nRemain there: StayThere\nThank you: PraiseMe\nPause: StayThere\nCome here: ComeHere\nGood boy: PraiseMe\nSilence K9: StopListening\nWhat is your name: QuestionMe\nWalk behind me: FollowMe\nWalkies: FollowMe\nChange direction: TurnAbout\nQuiet: StopListening\nStop listening: StopListening\nTime for a walk: FollowMe\nWhy are you made of metal: QuestionMe\nTime to sleep: StopListening\nWhere is the door: QuestionMe\nWould you like to play a game of chess?: PlayChesss\n"
        self.conv_model = "text-curie-001"
        self.intent_model = "text-curie-001"

    def robot_response(self, command:str) -> Tuple[str,str]:
        """Returns intent and response and stores conversation details between calls"""
        # Determine intent of command
        intent_obj = openai.Completion.create(
            model = self.intent_model,
            prompt=self.intent_training + command + ":",
            temperature=0,
            max_tokens=10,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
            stop=["\n"]
            )
        intent = intent_obj['choices'][0]['text']
        intent = ''.join(intent.split()) # remove spaces, newlines etc
        # Now determine response for K9 to speak
        response_obj = openai.Completion.create(
            model = self.conv_model,
            prompt = self.base_story + self.conversation + command + self.k9_suffix,
            temperature = 1,
            max_tokens = 40,
            top_p = 1,
            frequency_penalty = 0.0,
            presence_penalty = 0.0,
            stop=["Human:"]
            # logit_bias = {35191:5, 2533:5, 876:5, 32863:5, 18254:5, 9866:5}
        )
        response = response_obj['choices'][0]['text']
        response =  response.strip('\n')

        # print("K9: " + response)
        self.conversation = self.conversation + command + self.k9_suffix + response + self.human_suffix
        # print(conversation)
        length =  self.conversation.count('\n')
        if length >= 20:
            self.conversation = self.conversation.split("\n",2)[2]
        return intent,response