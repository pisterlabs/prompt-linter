import openai
import numpy as np


question = 'what is the purpose of life?'

cards = ["The Fool",    "The Magician",    "The High Priestess",    "The Empress",    "The Emperor",    "The Hierophant",    "The Lovers",    "The Chariot",    "Strength",    "The Hermit",    "Wheel of Fortune",    "Justice",    "The Hanged Man",    "Death",    "Temperance",    "The Devil",    "The Tower",    "The Star",    "The Moon",    "The Sun",    "Judgement",    "The World",    "Ace of Wands",    "Two of Wands",    "Three of Wands",    "Four of Wands",    "Five of Wands",    "Six of Wands",    "Seven of Wands",    "Eight of Wands",    "Nine of Wands",    "Ten of Wands",    "Page of Wands",    "Knight of Wands",    "Queen of Wands",    "King of Wands",    "Ace of Cups",    "Two of Cups",    "Three of Cups",    "Four of Cups",    "Five of Cups",    "Six of Cups",    "Seven of Cups",    "Eight of Cups",    "Nine of Cups",    "Ten of Cups",    "Page of Cups",    "Knight of Cups",    "Queen of Cups",    "King of Cups",    "Ace of Swords",    "Two of Swords",    "Three of Swords",    "Four of Swords",    "Five of Swords",    "Six of Swords",    "Seven of Swords",    "Eight of Swords",    "Nine of Swords",    "Ten of Swords",    "Page of Swords",    "Knight of Swords",    "Queen of Swords",    "King of Swords",    "Ace of Pentacles",    "Two of Pentacles",    "Three of Pentacles",    "Four of Pentacles",    "Five of Pentacles",    "Six of Pentacles",    "Seven of Pentacles",    "Eight of Pentacles",    "Nine of Pentacles",    "Ten of Pentacles",    "Page of Pentacles",    "Knight of Pentacles",    "Queen of Pentacles",    "King of Pentacles"]

openai.api_key = open(r'misc/apikey.txt', 'r').read().strip("\n")

mainPrompt =  ('You are a tarot reader and your job is to return a tarot reading given 3 cards and a question. Your response should always include 5'
                'text blocks, the first is where you pretend that you were the ones that drew the card, this first block should not be very long, the next 3 text blocks should be a paragraph for each card describing the meaning, and the '
                'last should be a summary paragraph where you discuss what the combination of cards could mean as well as vaguely addressing the users question, it is fairly important that '
               'you discuss what different combinations mean. '
                'Your reading should not include any disclaimers or acknowledgments of instructions and should only include the reading. It should also be emphasized that you '
                'are emulating a human tarot reader so you should be confident in your wording. Lastly, each text block should be seperated by a double newline character, do not include labels for anything. The most important thing to keep in mind is that there should be a total of 5 text blocks each seperated by a double newline this means that should be a total of 4 occurances of the double newline sperator'
               'also make sure there is always a double newline seperater between the first text block annoucning which cards were drawn and the first cards description')



class Reader:
    def __init__(self, question):
        #super().__init__()
        self.history = [{"role": "user", "content": f"{mainPrompt}"}, {"role": "assistant", "content": "ok, I will provide tarot readings based on those insturctions and the given cards and question"}]
        self.question = question
        self.chosen = self.getCards()
        print(self.chosen, 'CHOSEN')
        self.readingPrompt = ({"role":"user", "content":f"following the instructions from the previous message, provide a reading for the following cards: {self.chosen}, and questions: {self.question}"})
        self.blocks = self.readCards()
        #self.blocks = ['I sense a powerful and transformative energy coming through these cards. As if fate has a hand in what is being revealed. ', 'The first card is Death. This card often evokes fear but instead it represents a powerful message of transformation and change. It signifies the end of a cycle, the shedding of old patterns and habits, and the coming of something new. It is a card of rebirth and letting go of what no longer serves us. ', 'The second card is the Two of Pentacles reversed. This card suggests that the querent is struggling to find balance in their life right now. Perhaps they are overburdened with responsibilities or are finding it difficult to juggle various projects. It may be time to reassess their priorities and make space for the changes that the Death card has brought. ', 'The third card is The World, which indicates that there is a sense of completion and wholeness in the air. The querent is being called to embrace their inner strength and knowledge, and use it to help them transition to the next phase of their life successfully. They are coming to the end of a major journey or chapter in their life, and it is time to celebrate their achievements and move forward with confidence. ', 'Overall, this reading suggests that the querent is going through a major transformation and that the universe is calling for a re-evaluation of their priorities. It may be challenging to adapt to the changes that are coming, but the completion of this cycle will bring about a sense of accomplishment and wholeness. Trust the process, and embrace what is to come.']

    def getCards(self):
        chosen = []
        for i in range(3):
            print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
            reverse = np.random.choice([0, 1])
            print('Revers - ', reverse)
            if reverse == 0:
                reverse = False
            elif reverse == 1:
                reverse = True

            card = np.random.randint(0, 77)
            print('card - ', card)

            if card not in chosen:
                if reverse:
                    card = np.random.randint(0, 77)
                    card = cards[card]
                    card = card + ' Reverse'
                    chosen.append(card)
                else:
                    card = cards[card]
                    chosen.append(card)


            elif card in chosen:
                if reverse:
                    card = np.random.randint(0, 77)
                    card = cards[card]
                    card = card + ' Reverse'
                    chosen.append(card)
                else:
                    card = 77 - card
                    card = cards[card]
                    chosen.append(card)
        return chosen

    def readCards(self):
        self.history.append(self.readingPrompt)
        returned = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=self.history
        )
        text = returned.choices[0].message.content
        blocks = text.split('\n\n')
        print(blocks)
        return blocks

