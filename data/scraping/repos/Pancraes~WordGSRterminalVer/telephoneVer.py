import cohere
import os

answers=[]

def checkWord(s):
	if len(s.split()) == 1:
		return True
	return False

os.system('cls||clear')
print("\n--- --- --- Telephone Version --- --- ---")
rules = input("\n\nWould you like to see the rules? (y/n) ")
  
if rules == "y":
  print("\n\n1) A player will enter two words/names(make sure to capitalize the name).\n\n2) All the players will choose a difficulty: easy, medium or hard. \n\n3) The first player will enter two words, which will generate a sentence. Depending on the difficulty, the sentence generated from those two words will be more complex.\n\n4) The first player will pass their device to another player. The player will then try to guess which two words were entered by the first player through the sentence generated, and so on. \n\n5) This will go on until you reach the final player, where you can see how different your words have gotten from the original words!")
  input("\nEnter any character to continue: ")

os.system('cls||clear')

while True: 
  numPlayers = int(input("How many players are playing? "))
  if numPlayers >= 3 and numPlayers <=8:
    break
  else:
    print ("Please enter a valid number (between 3 and 8 inclusive)")
    continue

os.system('cls||clear')

for player in range(numPlayers-1): #last player does not need prompt
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
  print(f"Sentence:{response.generations[0].text}\n")

print('Player', numPlayers, '\n')
while True: # outside bc we dont want it to show another generated sentence
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

print('Original Words')
print(f'\nPlayer 1: {answers[0][0]}, {answers[0][1]}')

print('\nYour guesses:')
for i in range(1, numPlayers):
  print(f'Player {i+1}: {answers[i][0]}, {answers[i][1]}')
input("\nPress any key to continue: ")