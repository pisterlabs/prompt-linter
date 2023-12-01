import tkinter as tk
from PIL import ImageTk, Image
import openai
import urllib.request
from tkinter import messagebox
import random

#tells us which label row is under use
current_row = 1
left_output = list()
right_output = list()
conditions = set()
def cursor_finder():
    
    global current_row
    empty_label_column=0
    
    for i in range(1,6): # 1,2,3,4,5
        if eval(f'square{current_row}{i}').cget("text")!='':
            continue
        else:
            empty_label_column=i
            return [current_row,empty_label_column]
    
    

def button_func(letter:str):
    
    cursor_position=cursor_finder()

    eval(f'square{cursor_position[0]}{cursor_position[1]}').configure(text=letter)
    eval(f'squarer{cursor_position[0]}{cursor_position[1]}').configure(text=letter)


def enter_button():
    #this won't work
    #check = cursor_finder()
    #if check[1]!=0:
    #    messagebox.showerror('','Please enter a 5 letter word')
    global current_row
    f = open("dictofwords.txt","r")
    
    dict_of_words = f.read()
    dict_of_words= eval(dict_of_words)

    def tryexcept():
        try:
            dict_of_words[guess_word.lower()]
            return False
        except KeyError:
            return True
            

    guess_word=str()
    for i in range(1,6):
        guess_word+=eval(f'square{current_row}{i}').cget("text")
    
    if len(guess_word)<5:
        messagebox.showerror('','Please enter a 5 letter word')
        for i in range(5,0,-1):
            eval(f'square{current_row}{i}').configure(text='')
            eval(f'squarer{current_row}{i}').configure(text='')
        
    elif tryexcept():
        messagebox.showerror('','This is not a word')
        for i in range(5,0,-1):
            eval(f'square{current_row}{i}').configure(text='')
            eval(f'squarer{current_row}{i}').configure(text='')
        
    else:

        global right_output, left_output

        def checker(guess_word):
            
            global hidden_word_1, hidden_word_2
            
            
            checker_output_1=[]
            checker_output_2=[]
            print(hidden_word_1,hidden_word_2)
            for i,j in zip(list(hidden_word_1), list(guess_word)):
                
                if i==j:
                    checker_output_1.append('g')
                elif j in list(hidden_word_1):
                    checker_output_1.append('y')
                else:
                    checker_output_1.append('gr')
                
            
            for i,j in zip(list(hidden_word_2), list(guess_word)):
                
                if i==j:
                    checker_output_2.append('g')
                elif j in list(hidden_word_2):
                    checker_output_2.append('y')
                else:
                    checker_output_2.append('gr')
                
            
            return checker_output_1, checker_output_2

        left_output, right_output = checker(guess_word)
        
        global conditions
        if left_output == ['g','g','g','g','g']:
            conditions.add('L')
        if right_output == ['g','g','g','g','g']:
            conditions.add('R')
        if len(conditions) == 2:
            messagebox.showinfo('Congratulations','Congratulations, you won!')
            dis.quit()
        if current_row == 7:
            if len(conditions)!=2:
                messagebox.showinfo('Ran out of tries',f'The words were {hidden_word_1} and {hidden_word_2}')
                dis.quit()
            



        #Update images of buttons
        for i,j,l,k in zip(guess_word, left_output, right_output,range(5)):
            #D:\Python_Mini_Project\b_gr_gr.png
            #imageb = Image.open('D:\\Python_Mini_Project\\'+f'b_{j}_{l}.png')
            #imagel = Image.open('D:\\Python_Mini_Project\\'+f'b_{j}_{j}.png')
            #imager = Image.open('D:\\Python_Mini_Project\\'+f'b_{l}_{l}.png')
            #eval(f'button{i.lower()}').configure(image=ImageTk.PhotoImage(imageb))
            #eval(f'square{current_row}{k+1}').configure(image = ImageTk.PhotoImage(imagel))
            #eval(f'squarer{current_row}{k+1}').configure(image = ImageTk.PhotoImage(imager))
            
            dark_grey = "#777b7d"
            yellow = "#c9b458"
            green = "#6aaa64"
            def colorgiver(x):
                if x == 'g':
                    
                    return "#6aaa64"
                elif x == 'y':
                    
                    return "#c9b458"
                else:
                    
                    return "#777b7d"


            eval(f'square{current_row}{k+1}').configure(bg = colorgiver(j))
            eval(f'squarer{current_row}{k+1}').configure(bg = colorgiver(l))
            
        print(current_row)
        current_row+=1



def backspace_button():

    cursor_pos=cursor_finder()
    #accessing the column before the next empty column
    

    eval(f'square{cursor_pos[0]}{cursor_pos[1]-1}').configure(text='')
    eval(f'squarer{cursor_pos[0]}{cursor_pos[1]-1}').configure(text = '')


def target_word():
    
    f = open("targets.txt","r")
    
    list_target_words = f.read()
    list_target_words = eval(list_target_words)
    
    hidden_word_1 = random.choice(list_target_words)

    list_target_words.remove(hidden_word_1)

    hidden_word_2 = random.choice(list_target_words)

    return hidden_word_1.upper(), hidden_word_2.upper()

def api_getter(prompt):
    
    openai.api_key = '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'
    
    x=openai.Image.create(
        prompt=prompt,
        n=1,
        size='512x512'
    )
    
    urllib.request.urlretrieve(x['data'][0]['url'],"gfg.png")
    
    img = Image.open("gfg.png")
    
    return img

dis=tk.Tk('[ai_rdle]')
dis.configure(bg='black')

frame_container_left=tk.Frame(dis,background='black')
frame_container_left.grid(row=0,column=0)



square11 = tk.Label(frame_container_left, font=("Arial", 10, "bold"), text="", width=8, height=4, borderwidth=1, relief="ridge", anchor="center") 
square11.grid(row=0, column=0)
square21 = tk.Label(frame_container_left, font=("Arial", 10, "bold"), text="", width=8, height=4, borderwidth=1, relief="ridge", anchor="center")
square21.grid(row=1, column=0)
square31 = tk.Label(frame_container_left, font=("Arial", 10, "bold"), text="", width=8, height=4, borderwidth=1, relief="ridge", anchor="center")
square31.grid(row=2, column=0)
square41 = tk.Label(frame_container_left, font=("Arial", 10, "bold"), text="", width=8, height=4, borderwidth=1, relief="ridge", anchor="center")
square41.grid(row=3, column=0)
square51 = tk.Label(frame_container_left, font=("Arial", 10, "bold"), text="", width=8, height=4, borderwidth=1, relief="ridge", anchor="center")
square51.grid(row=4, column=0)
square61 = tk.Label(frame_container_left, font=("Arial", 10, "bold"), text="", width=8, height=4, borderwidth=1, relief="ridge", anchor="center")
square61.grid(row=5, column=0)
square71 = tk.Label(frame_container_left, font=("Arial", 10, "bold"), text="", width=8, height=4, borderwidth=1, relief="ridge", anchor="center")
square71.grid(row=6, column=0)
square12 = tk.Label(frame_container_left, font=("Arial", 10, "bold"), text="", width=8, height=4, borderwidth=1, relief="ridge", anchor="center")
square12.grid(row=0, column=1)
square22 = tk.Label(frame_container_left, font=("Arial", 10, "bold"), text="", width=8, height=4, borderwidth=1, relief="ridge", anchor="center")
square22.grid(row=1, column=1)
square32 = tk.Label(frame_container_left, font=("Arial", 10, "bold"), text="", width=8, height=4, borderwidth=1, relief="ridge", anchor="center")
square32.grid(row=2, column=1)
square42 = tk.Label(frame_container_left, font=("Arial", 10, "bold"), text="", width=8, height=4, borderwidth=1, relief="ridge", anchor="center")
square42.grid(row=3, column=1)
square52 = tk.Label(frame_container_left, font=("Arial", 10, "bold"), text="", width=8, height=4, borderwidth=1, relief="ridge", anchor="center")
square52.grid(row=4, column=1)
square62 = tk.Label(frame_container_left, font=("Arial", 10, "bold"), text="", width=8, height=4, borderwidth=1, relief="ridge", anchor="center")
square62.grid(row=5, column=1)
square72 = tk.Label(frame_container_left, font=("Arial", 10, "bold"), text="", width=8, height=4, borderwidth=1, relief="ridge", anchor="center")
square72.grid(row=6, column=1)
square13 = tk.Label(frame_container_left, font=("Arial", 10, "bold"), text="", width=8, height=4, borderwidth=1, relief="ridge", anchor="center")
square13.grid(row=0, column=2)
square23 = tk.Label(frame_container_left, font=("Arial", 10, "bold"), text="", width=8, height=4, borderwidth=1, relief="ridge", anchor="center")
square23.grid(row=1, column=2)
square33 = tk.Label(frame_container_left, font=("Arial", 10, "bold"), text="", width=8, height=4, borderwidth=1, relief="ridge", anchor="center")
square33.grid(row=2, column=2)
square43 = tk.Label(frame_container_left, font=("Arial", 10, "bold"), text="", width=8, height=4, borderwidth=1, relief="ridge", anchor="center")
square43.grid(row=3, column=2)
square53 = tk.Label(frame_container_left, font=("Arial", 10, "bold"), text="", width=8, height=4, borderwidth=1, relief="ridge", anchor="center")
square53.grid(row=4, column=2)
square63 = tk.Label(frame_container_left, font=("Arial", 10, "bold"), text="", width=8, height=4, borderwidth=1, relief="ridge", anchor="center")
square63.grid(row=5, column=2)
square73 = tk.Label(frame_container_left, font=("Arial", 10, "bold"), text="", width=8, height=4, borderwidth=1, relief="ridge", anchor="center")
square73.grid(row=6, column=2)
square14 = tk.Label(frame_container_left, font=("Arial", 10, "bold"), text="", width=8, height=4, borderwidth=1, relief="ridge", anchor="center")
square14.grid(row=0, column=3)
square24 = tk.Label(frame_container_left, font=("Arial", 10, "bold"), text="", width=8, height=4, borderwidth=1, relief="ridge", anchor="center")
square24.grid(row=1, column=3)
square34 = tk.Label(frame_container_left, font=("Arial", 10, "bold"), text="", width=8, height=4, borderwidth=1, relief="ridge", anchor="center")
square34.grid(row=2, column=3)
square44 = tk.Label(frame_container_left, font=("Arial", 10, "bold"), text="", width=8, height=4, borderwidth=1, relief="ridge", anchor="center")
square44.grid(row=3, column=3)
square54 = tk.Label(frame_container_left, font=("Arial", 10, "bold"), text="", width=8, height=4, borderwidth=1, relief="ridge", anchor="center")
square54.grid(row=4, column=3)
square64 = tk.Label(frame_container_left, font=("Arial", 10, "bold"), text="", width=8, height=4, borderwidth=1, relief="ridge", anchor="center")
square64.grid(row=5, column=3)
square74 = tk.Label(frame_container_left, font=("Arial", 10, "bold"), text="", width=8, height=4, borderwidth=1, relief="ridge", anchor="center")
square74.grid(row=6, column=3)
square15 = tk.Label(frame_container_left, font=("Arial", 10, "bold"), text="", width=8, height=4, borderwidth=1, relief="ridge", anchor="center")
square15.grid(row=0, column=4)
square25 = tk.Label(frame_container_left, font=("Arial", 10, "bold"), text="", width=8, height=4, borderwidth=1, relief="ridge", anchor="center")
square25.grid(row=1, column=4)
square35 = tk.Label(frame_container_left, font=("Arial", 10, "bold"), text="", width=8, height=4, borderwidth=1, relief="ridge", anchor="center")
square35.grid(row=2, column=4)
square45 = tk.Label(frame_container_left, font=("Arial", 10, "bold"), text="", width=8, height=4, borderwidth=1, relief="ridge", anchor="center")
square45.grid(row=3, column=4)
square55 = tk.Label(frame_container_left, font=("Arial", 10, "bold"), text="", width=8, height=4, borderwidth=1, relief="ridge", anchor="center")
square55.grid(row=4, column=4)
square65 = tk.Label(frame_container_left, font=("Arial", 10, "bold"), text="", width=8, height=4, borderwidth=1, relief="ridge", anchor="center")
square65.grid(row=5, column=4)
square75 = tk.Label(frame_container_left, font=("Arial", 10, "bold"), text="", width=8, height=4, borderwidth=1, relief="ridge", anchor="center")
square75.grid(row=6, column=4)



hidden_word_1, hidden_word_2 = target_word()


img=ImageTk.PhotoImage(api_getter(hidden_word_1+ " and " +hidden_word_2))
label_api=tk.Label(dis, image=img, borderwidth=0)
label_api.grid(row=0, column=1)

frame_container_right=tk.Frame(dis,background='black')
frame_container_right.grid(row=0,column=2)




squarer11 = tk.Label(frame_container_right, font=("Arial", 10, "bold"), text="", width=8, height=4, borderwidth=1, relief="ridge", anchor="center") 
squarer11.grid(row=0, column=0)
squarer21 = tk.Label(frame_container_right, font=("Arial", 10, "bold"), text="", width=8, height=4, borderwidth=1, relief="ridge", anchor="center")
squarer21.grid(row=1, column=0)
squarer31 = tk.Label(frame_container_right, font=("Arial", 10, "bold"), text="", width=8, height=4, borderwidth=1, relief="ridge", anchor="center")
squarer31.grid(row=2, column=0)
squarer41 = tk.Label(frame_container_right, font=("Arial", 10, "bold"), text="", width=8, height=4, borderwidth=1, relief="ridge", anchor="center")
squarer41.grid(row=3, column=0)
squarer51 = tk.Label(frame_container_right, font=("Arial", 10, "bold"), text="", width=8, height=4, borderwidth=1, relief="ridge", anchor="center")
squarer51.grid(row=4, column=0)
squarer61 = tk.Label(frame_container_right, font=("Arial", 10, "bold"), text="", width=8, height=4, borderwidth=1, relief="ridge", anchor="center")
squarer61.grid(row=5, column=0)
squarer71 = tk.Label(frame_container_right, font=("Arial", 10, "bold"), text="", width=8, height=4, borderwidth=1, relief="ridge", anchor="center")
squarer71.grid(row=6, column=0)
squarer12 = tk.Label(frame_container_right, font=("Arial", 10, "bold"), text="", width=8, height=4, borderwidth=1, relief="ridge", anchor="center")
squarer12.grid(row=0, column=1)
squarer22 = tk.Label(frame_container_right, font=("Arial", 10, "bold"), text="", width=8, height=4, borderwidth=1, relief="ridge", anchor="center")
squarer22.grid(row=1, column=1)
squarer32 = tk.Label(frame_container_right, font=("Arial", 10, "bold"), text="", width=8, height=4, borderwidth=1, relief="ridge", anchor="center")
squarer32.grid(row=2, column=1)
squarer42 = tk.Label(frame_container_right, font=("Arial", 10, "bold"), text="", width=8, height=4, borderwidth=1, relief="ridge", anchor="center")
squarer42.grid(row=3, column=1)
squarer52 = tk.Label(frame_container_right, font=("Arial", 10, "bold"), text="", width=8, height=4, borderwidth=1, relief="ridge", anchor="center")
squarer52.grid(row=4, column=1)
squarer62 = tk.Label(frame_container_right, font=("Arial", 10, "bold"), text="", width=8, height=4, borderwidth=1, relief="ridge", anchor="center")
squarer62.grid(row=5, column=1)
squarer72 = tk.Label(frame_container_right, font=("Arial", 10, "bold"), text="", width=8, height=4, borderwidth=1, relief="ridge", anchor="center")
squarer72.grid(row=6, column=1)
squarer13 = tk.Label(frame_container_right, font=("Arial", 10, "bold"), text="", width=8, height=4, borderwidth=1, relief="ridge", anchor="center")
squarer13.grid(row=0, column=2)
squarer23 = tk.Label(frame_container_right, font=("Arial", 10, "bold"), text="", width=8, height=4, borderwidth=1, relief="ridge", anchor="center")
squarer23.grid(row=1, column=2)
squarer33 = tk.Label(frame_container_right, font=("Arial", 10, "bold"), text="", width=8, height=4, borderwidth=1, relief="ridge", anchor="center")
squarer33.grid(row=2, column=2)
squarer43 = tk.Label(frame_container_right, font=("Arial", 10, "bold"), text="", width=8, height=4, borderwidth=1, relief="ridge", anchor="center")
squarer43.grid(row=3, column=2)
squarer53 = tk.Label(frame_container_right, font=("Arial", 10, "bold"), text="", width=8, height=4, borderwidth=1, relief="ridge", anchor="center")
squarer53.grid(row=4, column=2)
squarer63 = tk.Label(frame_container_right, font=("Arial", 10, "bold"), text="", width=8, height=4, borderwidth=1, relief="ridge", anchor="center")
squarer63.grid(row=5, column=2)
squarer73 = tk.Label(frame_container_right, font=("Arial", 10, "bold"), text="", width=8, height=4, borderwidth=1, relief="ridge", anchor="center")
squarer73.grid(row=6, column=2)
squarer14 = tk.Label(frame_container_right, font=("Arial", 10, "bold"), text="", width=8, height=4, borderwidth=1, relief="ridge", anchor="center")
squarer14.grid(row=0, column=3)
squarer24 = tk.Label(frame_container_right, font=("Arial", 10, "bold"), text="", width=8, height=4, borderwidth=1, relief="ridge", anchor="center")
squarer24.grid(row=1, column=3)
squarer34 = tk.Label(frame_container_right, font=("Arial", 10, "bold"), text="", width=8, height=4, borderwidth=1, relief="ridge", anchor="center")
squarer34.grid(row=2, column=3)
squarer44 = tk.Label(frame_container_right, font=("Arial", 10, "bold"), text="", width=8, height=4, borderwidth=1, relief="ridge", anchor="center")
squarer44.grid(row=3, column=3)
squarer54 = tk.Label(frame_container_right, font=("Arial", 10, "bold"), text="", width=8, height=4, borderwidth=1, relief="ridge", anchor="center")
squarer54.grid(row=4, column=3)
squarer64 = tk.Label(frame_container_right, font=("Arial", 10, "bold"), text="", width=8, height=4, borderwidth=1, relief="ridge", anchor="center")
squarer64.grid(row=5, column=3)
squarer74 = tk.Label(frame_container_right, font=("Arial", 10, "bold"), text="", width=8, height=4, borderwidth=1, relief="ridge", anchor="center")
squarer74.grid(row=6, column=3)
squarer15 = tk.Label(frame_container_right, font=("Arial", 10, "bold"), text="", width=8, height=4, borderwidth=1, relief="ridge", anchor="center")
squarer15.grid(row=0, column=4)
squarer25 = tk.Label(frame_container_right, font=("Arial", 10, "bold"), text="", width=8, height=4, borderwidth=1, relief="ridge", anchor="center")
squarer25.grid(row=1, column=4)
squarer35 = tk.Label(frame_container_right, font=("Arial", 10, "bold"), text="", width=8, height=4, borderwidth=1, relief="ridge", anchor="center")
squarer35.grid(row=2, column=4)
squarer45 = tk.Label(frame_container_right, font=("Arial", 10, "bold"), text="", width=8, height=4, borderwidth=1, relief="ridge", anchor="center")
squarer45.grid(row=3, column=4)
squarer55 = tk.Label(frame_container_right, font=("Arial", 10, "bold"), text="", width=8, height=4, borderwidth=1, relief="ridge", anchor="center")
squarer55.grid(row=4, column=4)
squarer65 = tk.Label(frame_container_right, font=("Arial", 10, "bold"), text="", width=8, height=4, borderwidth=1, relief="ridge", anchor="center")
squarer65.grid(row=5, column=4)
squarer75 = tk.Label(frame_container_right, font=("Arial", 10, "bold"), text="", width=8, height=4, borderwidth=1, relief="ridge", anchor="center")
squarer75.grid(row=6, column=4)



frame_keyboard_1=tk.Frame(dis, background='black')
frame_keyboard_1.grid(row=1,column=1)



buttonq=tk.Button(frame_keyboard_1, text='Q', compound ="center", font=("Arial",20), fg="white",bg = "black", command=lambda : button_func("Q"))
buttonq.grid(row = 1,column = 0)
buttonw=tk.Button(frame_keyboard_1, text='W', compound ="center", font=("Arial",20), fg="white",bg = "black", command=lambda : button_func("W"))
buttonw.grid(row = 1,column = 1)
buttone=tk.Button(frame_keyboard_1, text='E', compound ="center", font=("Arial",20), fg="white",bg = "black", command=lambda : button_func("E"))
buttone.grid(row = 1,column = 2)
buttonr=tk.Button(frame_keyboard_1, text='R', compound ="center", font=("Arial",20), fg="white",bg = "black", command=lambda : button_func("R"))
buttonr.grid(row = 1,column = 3)
buttont=tk.Button(frame_keyboard_1, text='T', compound ="center", font=("Arial",20), fg="white",bg = "black", command=lambda : button_func("T"))
buttont.grid(row = 1,column = 4)
buttony=tk.Button(frame_keyboard_1, text='Y', compound ="center", font=("Arial",20), fg="white",bg = "black", command=lambda : button_func("Y"))
buttony.grid(row = 1,column = 5)
buttonu=tk.Button(frame_keyboard_1, text='U', compound ="center", font=("Arial",20), fg="white",bg = "black", command=lambda : button_func("U"))
buttonu.grid(row = 1,column = 6)
buttoni=tk.Button(frame_keyboard_1, text='I', compound ="center", font=("Arial",20), fg="white",bg = "black", command=lambda : button_func("I"))
buttoni.grid(row = 1,column = 7)
buttono=tk.Button(frame_keyboard_1, text='O', compound ="center", font=("Arial",20), fg="white",bg = "black", command=lambda : button_func("O"))
buttono.grid(row = 1,column = 8)
buttonp=tk.Button(frame_keyboard_1, text='P', compound ="center", font=("Arial",20), fg="white",bg = "black", command=lambda : button_func("P"))
buttonp.grid(row = 1,column = 9)



frame_keyboard_2=tk.Frame(dis,background='black')
frame_keyboard_2.grid(row=2, column=1)



buttona=tk.Button(frame_keyboard_2, text='A', compound ="center", font=("Arial",20), fg="white",bg = "black", command=lambda : button_func("A"))
buttona.grid(row = 2,column = 0)
buttons=tk.Button(frame_keyboard_2, text='S', compound ="center", font=("Arial",20), fg="white",bg = "black", command=lambda : button_func("S"))
buttons.grid(row = 2,column = 1)
buttond=tk.Button(frame_keyboard_2, text='D', compound ="center", font=("Arial",20), fg="white",bg = "black", command=lambda : button_func('D'))
buttond.grid(row = 2,column = 2)
buttonf=tk.Button(frame_keyboard_2, text='F', compound ="center", font=("Arial",20), fg="white",bg = "black", command=lambda : button_func('F'))
buttonf.grid(row = 2,column = 3)
buttong=tk.Button(frame_keyboard_2, text='G', compound ="center", font=("Arial",20), fg="white",bg = "black", command=lambda : button_func('G'))
buttong.grid(row = 2,column = 4)
buttonh=tk.Button(frame_keyboard_2, text='H', compound ="center", font=("Arial",20), fg="white",bg = "black", command=lambda : button_func('H'))
buttonh.grid(row = 2,column = 5)
buttonj=tk.Button(frame_keyboard_2, text='J', compound ="center", font=("Arial",20), fg="white",bg = "black", command=lambda : button_func('J'))
buttonj.grid(row = 2,column = 6)
buttonk=tk.Button(frame_keyboard_2, text='K', compound ="center", font=("Arial",20), fg="white",bg = "black", command=lambda : button_func('K'))
buttonk.grid(row = 2,column = 7)
buttonl=tk.Button(frame_keyboard_2, text='L', compound ="center", font=("Arial",20), fg="white",bg = "black", command=lambda : button_func('L'))
buttonl.grid(row = 2,column = 8)


frame_keyboard_3=tk.Frame(dis, background='black')
frame_keyboard_3.grid(row=3, column=1)



buttonenter = tk.Button(frame_keyboard_3, text='Enter', compound ="center", font=("Arial",20), fg="white",bg = "black", command=enter_button)
buttonenter.grid(row = 3,column = 0)
buttonz=tk.Button(frame_keyboard_3, text='Z', compound ="center", font=("Arial",20), fg="white",bg = "black", command=lambda : button_func('Z'))
buttonz.grid(row = 3,column = 1)
buttonx=tk.Button(frame_keyboard_3, text='X', compound ="center", font=("Arial",20), fg="white",bg = "black", command=lambda : button_func('X'))
buttonx.grid(row = 3,column = 2)
buttonc=tk.Button(frame_keyboard_3, text='C', compound ="center", font=("Arial",20), fg="white",bg = "black", command=lambda : button_func('C'))
buttonc.grid(row = 3,column = 3)
buttonv=tk.Button(frame_keyboard_3, text='V', compound ="center", font=("Arial",20), fg="white",bg = "black", command=lambda : button_func('V'))
buttonv.grid(row = 3,column = 4)
buttonb=tk.Button(frame_keyboard_3, text='B', compound ="center", font=("Arial",20), fg="white",bg = "black", command=lambda : button_func('B'))
buttonb.grid(row = 3,column = 5)
buttonn=tk.Button(frame_keyboard_3, text='N', compound ="center", font=("Arial",20), fg="white",bg = "black", command=lambda : button_func('N'))
buttonn.grid(row = 3,column = 6)
buttonm=tk.Button(frame_keyboard_3, text='M', compound ="center", font=("Arial",20), fg="white",bg = "black", command=lambda : button_func('M'))
buttonm.grid(row = 3,column = 7)
buttonback = tk.Button(frame_keyboard_3, text='Back', compound ="center", font=("Arial",20), fg="white",bg = "black", command = backspace_button)
buttonback.grid(row = 3,column = 8)



    
    
           























































dis.mainloop()
