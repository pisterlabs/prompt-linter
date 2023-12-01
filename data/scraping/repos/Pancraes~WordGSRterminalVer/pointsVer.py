import cohere
import os

answers=[]
playerPoint=[]
def checkWord(s):
	if len(s.split()) == 1:
		return True
	return False

os.system('cls||clear')
print("\n--- --- --- Points Version --- --- ---")
rules = input("\n\nWould you like to see the rules? (y/n) ")

if rules == "y":
  print("\n\n1) A player will enter two words/names(make sure to capitalize the name).\n\n2) The first player will enter two words, which will generate a sentence. Pass the device to the 2nd player \n\n3) The player will choose a difficulty, where the higher difficulty, the more complex the sentence will be.\n\n4) After the player enters their two guesses, they enter two more words for the next player, and so on. \n\n5) A round will end when all players have guessed, and at the end of every round, all players' points will be shown \n\n6) There will be 3 rounds to every game.")
  input("\nEnter any character to continue: ")

os.system('cls||clear')

while True: 
  numPlayers = int(input("How many players are playing? "))
  if numPlayers >= 3 and numPlayers <=8:
  	break
  else:
    print ("Please enter a valid number (between 3 and 8 inclusive)")
    continue

playerPoint = [0]*numPlayers
os.system('cls||clear')

for x in range(3): #3 rounds
  for player in range(numPlayers):
    print('Player', player+1, '\n')
    while True:
        word1 = input("Enter the first word: ")
        if checkWord(word1):
          break
        else:
          print("Please enter one word")
    while True:
      word2 = input("Enter the second word: ")
      if checkWord(word2):
        break
      else:
        print("Please enter one word")
    answers.append([word1, word2])

    os.system('cls||clear')
    print("Pass to the next player!\n")
      
    while True:
      difficulty = input("What difficulty would you like your sentence to be? (easy, medium, or hard): ")
      if difficulty == 'easy' or difficulty == 'medium' or difficulty == 'hard':        
        if difficulty == "easy":
          with open('easyPrompt.txt') as f:
            contents = f.read()
        elif difficulty == "medium":     
          with open('mediumPrompt.txt') as f:
            contents = f.read()
        elif difficulty == "hard":    
          with open('hardPrompt.txt') as f:
            contents = f.read()
      else:
        print("Please enter a difficulty")
        continue
      print('\nProcessing...')
      break

    p=f"{contents}\n\nUser input: {word1}, {word2}\nOutput:"
    
    co = cohere.Client('XH6WEkN6940HTNO4hl1517Hpl1pX7gW8hpS3RisW')
    
    response = co.generate(
      model='xlarge',
      prompt = p,
      max_tokens=100,
      temperature=0.2,
      stop_sequences=['.'],
      k=0,
      p=0)
    os.system('cls||clear')
    print(f"Sentence:{response.generations[0].text}")

    while True:
      guess1 = input("\nWhat do you think word number 1 is? ")
      if checkWord(guess1):
        break
      else:
        print("Please enter one word")
    
    while True:
      guess2 = input("\nWhat do you think word number 2 is? ")
      if checkWord(guess2):
        break
      else:
        print("Please enter one word")
    
  
    if (guess1 == word1 or guess1 == word2):
      if difficulty == "hard":
        playerPoint[player] += 3
      elif difficulty == "medium":
        playerPoint[player] += 2
      else:
        playerPoint[player] += 1
  
    if (guess2 == word1 or guess2 == word2):
      if difficulty == "hard":
        playerPoint[player] += 3
      elif difficulty == "medium":
        playerPoint[player] += 2
      else:
        playerPoint[player] += 1
      
    os.system('cls||clear')

  for i in range(numPlayers): #maybe make this into a leaderboard later
    print("Player", i+1, "has", playerPoint[i], "points")
  thing = input("Press any key to continue")