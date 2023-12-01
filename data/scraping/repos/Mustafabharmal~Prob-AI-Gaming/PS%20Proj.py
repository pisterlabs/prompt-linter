import tkinter as tk
from tkinter import *
from tkinter import *
from tkinter import Tk, PhotoImage, Label
from tkinter import messagebox
from PIL import ImageTk, Image
import random
import openai
from openai import Completion
import math
import os 

openai.api_key = "Your API Key"

root = tk.Tk()
root.state("zoomed")#"%dx%d" % (width, height)
root.wm_title('Home Page')

bg = PhotoImage(file = "/Users/mrperfect/Work/Project/Prob-AI-Gaming/Img/BGFinal.png")
img_label = tk.Label( root, image = bg)
img_label.place(x = 0, y = 0)


BRock=PhotoImage(file=r"/Users/mrperfect/Work/Project/Prob-AI-Gaming/Img/StoneF.png")
BRock = BRock.zoom(5) #with 250, I ended up running out of memory
BRock = BRock.subsample(10) 

BSessior=PhotoImage(file=r"/Users/mrperfect/Work/Project/Prob-AI-Gaming/Img/SessiorF.png")
BSessior = BSessior.zoom(5) #with 250, I ended up running out of memory
BSessior = BSessior.subsample(10) 

BPaper=PhotoImage(file=r"/Users/mrperfect/Work/Project/Prob-AI-Gaming/Img/PaperF.png")
BPaper = BPaper.zoom(5) #with 250, I ended up running out of memory
BPaper = BPaper.subsample(10) 

LDPlayer=PhotoImage(file=r"/Users/mrperfect/Work/Project/Prob-AI-Gaming/Img/default.jpg")
LDPlayer = LDPlayer.zoom(6) #with 250, I ended up running out of memory
LDPlayer = LDPlayer.subsample(10) 

LDComputer=PhotoImage(file=r"/Users/mrperfect/Work/Project/Prob-AI-Gaming/Img/defaultFlip.jpg")
LDComputer = LDComputer.zoom(6) #with 250, I ended up running out of memory
LDComputer = LDComputer.subsample(10) 

LPPlayer=PhotoImage(file=r"/Users/mrperfect/Work/Project/Prob-AI-Gaming/Img/paper.jpg")
LPPlayer = LPPlayer.zoom(5) #with 250, I ended up running out of memory
LPPlayer = LPPlayer.subsample(10) 

LPComputer=PhotoImage(file=r"/Users/mrperfect/Work/Project/Prob-AI-Gaming/Img/paperFlip.jpg")
LPComputer = LPComputer.zoom(5) #with 250, I ended up running out of memory
LPComputer = LPComputer.subsample(10) 

LSPlayer=PhotoImage(file=r"/Users/mrperfect/Work/Project/Prob-AI-Gaming/Img/scissor.jpg")
LSPlayer = LSPlayer.zoom(5) #with 250, I ended up running out of memory
LSPlayer = LSPlayer.subsample(10) 

LSComputer=PhotoImage(file=r"/Users/mrperfect/Work/Project/Prob-AI-Gaming/Img/scissorFlip.jpg")
LSComputer = LSComputer.zoom(5) #with 250, I ended up running out of memory
LSComputer = LSComputer.subsample(10) 

LRPlayer=PhotoImage(file=r"/Users/mrperfect/Work/Project/Prob-AI-Gaming/Img/rock.jpg")
LRPlayer = LRPlayer.zoom(5) #with 250, I ended up running out of memory
LRPlayer = LRPlayer.subsample(10) 

LRComputer=PhotoImage(file=r"/Users/mrperfect/Work/Project/Prob-AI-Gaming/Img/rockFlip.jpg")
LRComputer = LRComputer.zoom(5) #with 250, I ended up running out of memory
LRComputer = LRComputer.subsample(10) 
arr=[0,0]
def gpt():
    root.attributes('-topmost',False)
    Gpt = tk.Toplevel(root)
    Gpt.geometry("1000x1000")
    Gpt.title("OpenAI")
    Gpt['bg']='#0f1419'
    Gpt.attributes('-topmost',True)
    Gpt.resizable(0,0)
    # entry = tk.Entry(Gpt, font=('Arial', 14), bg='white', fg='black')
    LNote = tk.Label(Gpt,font=("Comic Sans MS", 20, 'bold'))
    LNote['text'] = 'WelCome To AI'
    LNote.place(x=400,y=20)
    
    text_Box = tk.Text(Gpt)
    scroll_Bar = tk.Scrollbar(text_Box, orient = tk.VERTICAL)
    text_Box.configure(font=("Verdana", 12), yscrollcommand = scroll_Bar.set)
    text_Box.place(x=100, y=650, width=800, height=100)
    scroll_Bar.config(command = text_Box.yview)
    scroll_Bar.pack(side=RIGHT, fill='y')

    # Create a text widget to display the conversation
    # conversation = tk.Text(root, font=('Arial', 14), bg='white', fg='black')
    conversation = tk.Text(Gpt)
    scroll_Bar = tk.Scrollbar(conversation, orient = tk.VERTICAL)
    conversation.configure(font=("Verdana", 12), yscrollcommand = scroll_Bar.set)
    conversation.place(x=100, y=100, width=800, height=500)
    scroll_Bar.config(command = conversation.yview)
    scroll_Bar.pack(side=RIGHT, fill='y')


    # Create a scrollbar for the conversation widget
    # scrollbar = tk.Scrollbar(root, orient='vertical', command=conversation.yview)
    # conversation['yscrollcommand'] = scrollbar.set

    # Create a function to clear the conversation widget
    def clear_conversation():
        conversation.delete('1.0', 'end')

    # Create a function to send a message to ChatGPT and display the response
    def send_message():
        # Get the message from the entry field
        message = text_Box.get('1.0',tk.END)

        # Clear the entry field
        text_Box.delete('1.0', tk.END)

        # Use the OpenAI API to get a response from ChatGPT
        response = Completion.create(
            engine="text-davinci-002",
            prompt=message,
            max_tokens=1024,
            temperature=0.5,
        )

        # Display the message and the response in the conversation widget
        conversation.insert(tk.END, f"\n>>>You:\n {message}")
        conversation.insert(tk.END, f"\n>>>Bot: {response.get('choices')[0].get('text')}\n")

    BSend = tk.Button(Gpt, bg='#e36414', fg='white',font=("Times new roman", 20, 'bold'),command=send_message)
    BSend['text'] = 'Send'
    BSend.place(x=100,y=800,width=800)

    BClear = tk.Button(Gpt, bg='#A20808', fg='white',font=("Times new roman", 20, 'bold'),command=clear_conversation)
    BClear['text'] = 'Clear'
    BClear.place(x=100,y=875,width=800)

    # Create the send button
    # send_button = tk.Button(root, text="Send", font=('Arial', 14), command=send_message, bg='white', fg='black')

    # Create the clear button
    # clear_button = tk.Button(root, text='Clear', command=clear_conversation, bg='white', fg='black')

    # Pack the widgets into the window
    # text_Box.pack()
    # send_button.pack()
    # clear_button.pack()
    # conversation.pack(side='left', fill='both', expand=True)
    # scrollbar.pack(side='right', fill='y')

def gam():
    root.attributes('-topmost',False)
    game = tk.Toplevel(root)
    game.geometry("800x1000")
    game.title("Game")
    # game['bg']='#CDCD9B'
    game.attributes('-topmost',True)
    game.resizable(0,0)
    
    
    
    LNote = tk.Label(game,font=("Comic Sans MS", 20, 'bold'))
    LNote['text'] = 'By How Many score a player will win:'
    LNote.place(x=50,y=20)
    
    Luser = tk.Label(game,font=("Comic Sans MS", 20, 'bold'))
    Luser['text'] = 'Response By You:'
    Luser.place(x=50,y=380)
    
    LComp = tk.Label(game,font=("Comic Sans MS", 20, 'bold'))
    LComp['text'] = 'Response By Comp:'
    LComp.place(x=470,y=380)
    
    PComp = tk.Label(game,font=("Comic Sans MS", 20, 'bold'))
    PComp['text'] = 'Probability of winning\n Computer:1/2'
    PComp.place(x=470,y=700)
    
    PPlay = tk.Label(game,font=("Comic Sans MS", 20, 'bold'))
    PPlay['text'] = 'Probability of winning\n Player:1/2'
    PPlay.place(x=50,y=700)
    
    SComp = tk.Label(game,font=("Comic Sans MS", 20, 'bold'))
    SComp['text'] = 'Score of Computer:0'
    SComp.place(x=470,y=800)
    
    SPlay = tk.Label(game,font=("Comic Sans MS", 20, 'bold'))
    SPlay['text'] = 'Score of Player:0'
    SPlay.place(x=50,y=800)
    
    Lvs = tk.Label(game,font=("Comic Sans MS", 28, 'bold'))
    Lvs['text'] = 'V/S'
    Lvs.place(x=370,y=550)
    
    text_Box = tk.Text(game)
    text_Box.configure(font=("Comic Sans MS", 20, 'bold'))
    text_Box.place(x=580, y=20, width=50, height=50)
    
    LPla = tk.Label(game,font=("Comic Sans MS", 20, 'bold'))
    # LPla['text'] = 'Player Selected:'
    LPla['image']=LDPlayer
    LPla.place(x=50,y=440)
    
    LCom = tk.Label(game,font=("Comic Sans MS", 20, 'bold'))
    # LCom['text'] = 'Player Selected:'
    LCom['image']=LDComputer
    LCom.place(x=470,y=440)
    
    LResult = tk.Label(game,font=("Comic Sans MS", 20, 'bold'))
    LResult['text'] = 'Result:'
    LResult.place(x=250,y=900)
    # game.messagebox.showinfo(" better Luck next time","Sorry Computer Won")
        
        # print(pno)
    
    def goo(player):
        select = [1, 2, 3]
        
        # Randomly selects option for computer
        computer = random.choice(select)
        pno=int(text_Box.get("1.0",END))

        # Setting image for player on canvas
        if player == 1:
            # Puts rock image on canvas
            # canvas.create_image(0, 100, anchor=NW, image=rock_p)
            LPla['image']=LRPlayer
        elif player == 2:
            
            # Puts paper image on canvas
            # canvas.create_image(0, 100, anchor=NW, image=paper_p)
            LPla['image']=LPPlayer
            
        else:
            
            # Puts scissor image on canvas
            # canvas.create_image(0, 100, anchor=NW, image=scissor_p)
            LPla['image']=LSPlayer
            

        # Setting image for computer on canvas
        if computer == 1:
            
            # Puts rock image on canvas
            # canvas.create_image(500, 100, anchor=NW, image=rock_c)
            LCom['image']=LRComputer
        elif computer == 2:
            LCom['image']=LPComputer
            
            # Puts paper image on canvas
            # canvas.create_image(500, 100, anchor=NW, image=paper_c)
        else:
            LCom['image']=LSComputer
            
            # Puts scissor image on canvas
            # canvas.create_image(500, 100, anchor=NW, image=scissor_c)
        
        # Obtaining result by comparison
        if player == computer:  # Case of DRAW
            print( 'Draw')
            # str='Result ='
            LResult.config(text=" ")
            LResult['text'] = 'Result: Draw'
            
            # SPlay.config(text=" ")
            # SPlay['text'] = 'Result: You Won'
            
        # Case of player's win
        elif (player == 1 and computer == 3):
            print( 'You won')
            LResult.config(text=" ")
            LResult['text'] = 'Result: You Won'
            
            arr[0]=arr[0]+1
            stro="Score of Player: " +str(arr[0])
            
            SPlay.config(text=" ")
            SPlay['text'] = stro
            
            stri="Probability of winning\n Player: "+str(arr[0]/pno);
            PPlay.config(text=" ")
            PPlay['text'] = stri
            
            stri="Probability of winning\n Computer: "+str(arr[1]/pno);
            PComp.config(text=" ")
            PComp['text'] = stri
            
            
        elif (player == 2 and computer == 1):
            print( 'You won')
            LResult.config(text=" ")
            LResult['text'] = 'Result: You Won'
            # arr[0]=arr[0]+1
            
            arr[0]=arr[0]+1
            stro="Score of Player: " +str(arr[0])
            
            SPlay.config(text=" ")
            SPlay['text'] = stro
            
            stri="Probability of winning\n Player: "+str(arr[0]/pno);
            PPlay.config(text=" ")
            PPlay['text'] = stri
            
            stri="Probability of winning\n Computer: "+str(arr[1]/pno);
            PComp.config(text=" ")
            PComp['text'] = stri
            
        elif (player == 3 and computer == 2):
            print( 'You won')
            LResult.config(text=" ")
            LResult['text'] = 'Result: You Won'
            # arr[0]=arr[0]+1
            arr[0]=arr[0]+1
            stro="Score of Player: " +str(arr[0])
            
            SPlay.config(text=" ")
            SPlay['text'] = stro
            
            stri="Probability of winning\n Player: "+str(arr[0]/pno);
            PPlay.config(text=" ")
            PPlay['text'] = stri
            
            stri="Probability of winning\n Computer: "+str(arr[1]/pno);
            PComp.config(text=" ")
            PComp['text'] = stri
        # Case of computer's win
        else:
            print('Computer won')
            LResult.config(text=" ")
            LResult['text'] = 'Result: Computer Won'
            arr[1]=arr[1]+1
            # arr[0]=arr[0]+1
            stro="Score of Computer: " +str(arr[1])
            
            SComp.config(text=" ")
            SComp['text'] = stro
            
            stri="Probability of winning\n Computer: "+str(arr[1]/pno);
            PComp.config(text=" ")
            PComp['text'] = stri
            
            stri="Probability of winning\n Player: "+str(arr[0]/pno);
            PPlay.config(text=" ")
            PPlay['text'] = stri
        
        if arr[1]==pno:
            print("Computer won")
            # game.attributes('-topmost',False)
            messagebox.showinfo(" better Luck next time","Try Again,Sorry Computer Won",parent=game)
            arr[0]=0
            arr[1]=0
            game.destroy()
        elif arr[0]==pno:
            print("Player won")
            # game.attributes('-topmost',False)
            messagebox.showinfo("Congratulations","Congratulation,You Won",parent=game)
            arr[1]=0
            arr[0]=0
            game.destroy()

    
    # bgp = PhotoImage(file = "Img/default.jpg")
    # img_label = tk.Label( game, image = bgp)
    # img_label.place(x = 50, y = 150)
    
    
    Sbutton = tk.Button(game, bg='#1A1A1A', fg='white',font=("Comic Sans MS", 20, 'bold'),image=BSessior,command=lambda: goo(3))
    Sbutton.place(x=450,y=200)
    Sbutton['state']='disable'
    
    PButton = tk.Button(game, bg='#1A1A1A', fg='white',font=("Comic Sans MS", 20, 'bold'),image=BPaper,command=lambda: goo(2))
    PButton.place(x=300,y=200)
    PButton['state']='disable'
    
    RButton = tk.Button(game, bg='#1A1A1A', fg='white',font=("Comic Sans MS", 20, 'bold'),image=BRock,comm=lambda: goo(1))
    RButton.place(x=150,y=200)
    RButton['state']='disable'
    def ply():
        text_Box['state']='disable'
        BPlay['state']='disable'
        Sbutton['state']='active'
        PButton['state']='active'
        RButton['state']='active'
    BPlay = tk.Button(game, bg='#1A1A1A', fg='white',font=("Comic Sans MS", 20, 'bold'),command=ply)
    BPlay['text'] = 'Play'
    BPlay.place(x=270,y=100,width=200,height=50)
answerVariableGlobal = ""
answerLabelForSquareRoot = ""   
def calc():
    root.attributes('-topmost',False)
    cal = tk.Toplevel(root)
    cal.geometry("590x740")
    cal.title("Calculator")
    # game['bg']='#CDCD9B'
    cal.attributes('-topmost',True)
    cal.resizable(0,0)
    
    answerEntryLabel = StringVar()
    Label(cal,font=('futura', 25, 'bold'), textvariable = answerEntryLabel, justify = LEFT,height=2, width=7).grid(columnspan=4, ipadx=120)
    #Label - Answer Label where the final answer would be shown after evaluating the expression entered.
    answerFinalLabel = StringVar()
    Label(cal,font=('futura', 25, 'bold'), textvariable = answerFinalLabel, justify = LEFT,height=2, width=7).grid(columnspan = 4 , ipadx=120)
    def changeAnswerEntryLabel(entry):
        #changeAnswerEntryLabel - adds the entry on click on a particular button 
        #to the answerVariableGlobal and also appends the entry to the answerEntryLabel
        global answerVariableGlobal
        global answerLabelForSquareRoot
        answerVariableGlobal = answerVariableGlobal + str(entry) #Adding entry on click of button to the answerVariableGlobal
        answerLabelForSquareRoot = answerVariableGlobal #Also modifying the answerVariableGlobal to the answerLabelForSquareRoot for future use  
        answerEntryLabel.set(answerVariableGlobal)#Showing each entry onto the answerEntryLabel of calculator by appending each entry to the previously entered values before evaluation or allClear
    def clearAnswerEntryLabel():
        #clears the answerEntryLabel and also clears answerVariableGlobal
        global answerVariableGlobal
        global answerLabelForSquareRoot
        answerLabelForSquareRoot = answerVariableGlobal
        answerVariableGlobal = ""
        answerEntryLabel.set(answerVariableGlobal)
    def evaluateSquareRoot():
        #evaluateSquareRoot - evaluates the expression present in
        #the answerLabelForSquareRoot for square cal of that value and 
        #returns that value to answerFinalLabel
        global answerVariableGlobal
        global answerLabelForSquareRoot
        # if answerVariableGlobal.__contains__('C'):
        #     print("C")
        
        try:
            sqrtAnswer = math.sqrt(eval(str(answerLabelForSquareRoot)))
            clearAnswerEntryLabel()
            answerFinalLabel.set(sqrtAnswer)
        except(ValueError,SyntaxError,TypeError, ZeroDivisionError):
            try:
                sqrtAnswer = math.sqrt(eval(str(answerVariableGlobal)))
                clearAnswerEntryLabel()
                answerFinalLabel.set(sqrtAnswer)
            except(ValueError,SyntaxError,TypeError, ZeroDivisionError):#ErrorHandling
                clearAnswerEntryLabel()
                answerFinalLabel.set("Error!")
    def evaluateAnswer():
        #evaluateAnswer - evaluates the expression present in
        #the answerVariableGlobal and returns the value to answerFinalLabel
        #also clearing the answerEntryLabel using clearAnswerEntryLabel()
        global answerVariableGlobal
        try:
            if answerVariableGlobal.__contains__('!'):
                answerVariableGlobal=answerVariableGlobal+str(0)
                ans=answerVariableGlobal.split('!')
                evaluatedValueAnswerLabelGlobal=str(math.factorial(int(ans[0])))
                print(ans[0])
                clearAnswerEntryLabel()
                answerFinalLabel.set(evaluatedValueAnswerLabelGlobal)
                
            elif answerVariableGlobal.__contains__('C'):
                ans=answerVariableGlobal.split('C')
                evaluatedValueAnswerLabelGlobal=str(math.comb(int(ans[0]),int(ans[1])))
                print(ans[0],ans[1])
                clearAnswerEntryLabel()
                answerFinalLabel.set(evaluatedValueAnswerLabelGlobal)
                
            elif answerVariableGlobal.__contains__('P'):
                # print(ans[0],ans[1])
                ans=answerVariableGlobal.split('P')
                print(ans[0],ans[1])
                evaluatedValueAnswerLabelGlobal=str(math.perm(int(ans[0]),int(ans[1])))
                
                clearAnswerEntryLabel()
                answerFinalLabel.set(evaluatedValueAnswerLabelGlobal)
            else:
                eval(answerVariableGlobal)
                evaluatedValueAnswerLabelGlobal = str(eval(answerVariableGlobal))
                clearAnswerEntryLabel()
                answerFinalLabel.set(evaluatedValueAnswerLabelGlobal)
        except(ValueError,SyntaxError,TypeError, ZeroDivisionError):#ErrorHandling
            clearAnswerEntryLabel()
            print(ValueError,SystemError,TypeError,ZeroDivisionError)
            answerFinalLabel.set("Error!")
    def allClear():
        #All Clear (AC) - clears out the current data,
        #and prepares the calculator to start a new calculation
        global answerVariableGlobal
        global answerLabelForSquareRoot
        answerVariableGlobal = ""
        answerLabelForSquareRoot = ""
        answerEntryLabel.set("")
        answerFinalLabel.set("")
    def createButton(txt,x,y):#Function used to create a button.
        Button(cal,font=('futura', 15, 'bold'),padx=16,pady=16,text = str(txt), command = lambda:changeAnswerEntryLabel(txt),height = 2, width=9).grid(row = x , column = y, sticky = E)

    # def fact():
    #     if answerEntryLabel.
    ###Buttons###
    #buttons list stores the button values to be incoroporated in the calculator for first 5 rows
    buttons = ['AC','√','%','/','7','8','9','*','4','5','6','-','1','2','3','+','','','.','']
    buttonsListTraversalCounter = 0 #buttonsListTraversalCounter is used to traverse across the buttons list  
    for i in range(3,8):
        for j in range(0,4):
            createButton(buttons[buttonsListTraversalCounter],i,j)
            buttonsListTraversalCounter = buttonsListTraversalCounter + 1
    Button(cal,font=('futura', 15, 'bold'),padx=16,pady=16, text = "√", command = lambda:evaluateSquareRoot(),height=2, width=9).grid(row = 3 , column = 1, sticky = E)#Button for SquareRoot
    Button(cal,font=('futura', 15, 'bold'),padx=16,pady=16, text = "AC", command = lambda:allClear(),height=2, width=9).grid(row = 3 , column = 0 , sticky = E)#Button for AC button - clear the workspace
    Button(cal,font=('futura', 15, 'bold'),padx=16,pady=16, text = "0", command = lambda:changeAnswerEntryLabel(0),height=2, width=21).grid(row = 7 , column = 0 , columnspan=2 , sticky = E)#Button for value 0
    Button(cal,font=('futura', 15, 'bold'),padx=16,pady=16, text = "=", command = lambda:evaluateAnswer(),height=2, width=9).grid(row = 7 , column = 3, sticky = E)#Button for "=" - final calc button  
    Button(cal,font=('futura', 15, 'bold'),padx=16,pady=16, text = "n!", command = lambda:changeAnswerEntryLabel('!'),height=2, width=21).grid(row = 8 , column = 0 ,columnspan=2 , sticky = E)#Button for value 0
    Button(cal,font=('futura', 15, 'bold'),padx=16,pady=16, text = "nCr", command = lambda:changeAnswerEntryLabel('C'),height=2, width=9).grid(row = 8 , column = 2, sticky = E)#Button for "=" - final calc button  
    Button(cal,font=('futura', 15, 'bold'),padx=16,pady=16, text = "nPr", command = lambda:changeAnswerEntryLabel('P'),height=2, width=9).grid(row = 8, column = 3, sticky = E)#Button for "=" - final calc button  

    #
BGame = tk.Button(root, bg='#e36414', fg='white',font=("Times new roman", 20, 'bold'),command=gam)
BGame['text'] = 'Game'
BGame.place(x=110,y=550,width=480)

BChat = tk.Button(root, bg='#e36414', fg='white',font=("Times new roman", 20, 'bold'),command=gpt)
BChat['text'] = 'AI'
BChat.place(x=723,y=550,width=480)

BCalc = tk.Button(root, bg='#e36414', fg='white',font=("Times new roman", 20, 'bold'),command=calc)
BCalc['text'] = 'Calculator'
BCalc.place(x=1336,y=550,width=480)

root.mainloop()
