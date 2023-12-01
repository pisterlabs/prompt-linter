import nltk
from nltk.corpus import stopwords
import pandas as pd
from nltk import FreqDist
import seaborn as sns
nltk.download('stopwords')
import openai
import joblib
from pathlib import Path
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix
from tkinter import *
from tkinter import filedialog
from tkinter import messagebox



model = joblib.load(Path('model_joblib'))
def sugesstion(arr):
    import os
    import openai
    openai.api_key = "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"

    response = openai.Completion.create(
      model="text-davinci-003",
      prompt="Give me 10 improvement tips based on following reviews for seller"+ arr[0]+arr[1]+arr[2]+arr[3]+arr[4],
      temperature=0.7,
      max_tokens=256,
      top_p=1,
      frequency_penalty=0,
      presence_penalty=0
    )
    sugesstions=response.choices[0].text.strip()
    return sugesstions




#action function
def Action(arr):
    ans = dict()
    
    for i in range(0,len(arr)):
        string= arr[i]
        temp=string.split()
        stop_words = stopwords.words('english')
        
        final_word_count=[]
        
        for j in range(0,len(temp)):
            if(temp[j] in stop_words):
                continue
            else:
                final_word_count.append(temp[j])           
        ans[i]=len(final_word_count) 
        
        final_word_count=[]
#key= index of the negetive review  value= count of suitable words
    
    sorted_suitable_word_count_with_index= sorted(ans.items(),key=lambda x:x[1],reverse=True)
    
    return sorted_suitable_word_count_with_index




#bad reviews seggregation


def unhappy(cols_as_np):
         
    #negetive reviews added
    bad_review_array=[]             

    for i in range(0,len(cols_as_np)):
    
        exp=[cols_as_np[i]]
        results=model.predict(exp)
        if(results[0]=='not happy'):
            bad_review_array.append(cols_as_np[i])
            
    sorted_suitable_word_count_with_index=Action(bad_review_array)      # calls action function to remove stopwords
    top5_reviews=[]
    for k in range(0,5):
        top5_reviews.append(bad_review_array[sorted_suitable_word_count_with_index[k][0]])
    #print(bad_review_array[sorted_suitable_word_count_with_index[k][0]])
    top5=sugesstion(top5_reviews)
    return top5


def input_file_organization(a):
    Reviewdata=[]    
    Reviewdata=a
    cols_as_np = Reviewdata['Reviews'].to_numpy()
    value_dictionary= unhappy(cols_as_np)
    return value_dictionary
    










'''
#input file organization
def input_file_organization():
    Reviewdata=[]
    import os
    path=input("Enter Path:")
    os.chdir(path)

    user_input=input("Enter File name:")
    a=user_input + ".csv"

    if os.path.exists(a):
        Reviewdata=pd.read_csv(a)
    
    else:
        print("File does not exists")
    

    cols_as_np = Reviewdata['Reviews'].to_numpy()
    value_dictionary= unhappy(cols_as_np)
    print(value_dictionary)
    

#input_file_organization()
'''


'''
from tkinter import *
window=Tk()
# add widgets here

window.title('Organization')
btn=Button(window, text="This is Button widget", fg='blue' ,command=input_file_organization)
btn.place(x=80, y=100)
window.geometry("300x200+10+20")
window.mainloop()
'''

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix
from tkinter import *
from tkinter import filedialog
from tkinter import messagebox
def train_classifier():
    # Get the path of the csv file
    csv_path = filedialog.askopenfilename(initialdir = "/",title = "Select a File",filetypes = (("CSV files","*.csv"),("all files","*.*")))
    if not csv_path:
        return

    # Load the csv file into a pandas DataFrame
    df = pd.read_csv(csv_path)
    ans= input_file_organization(df)
    # Preprocess the reviews by converting them to lowercase and removing punctuations and stopwords
    

    label3 = Label( root, text = "Here are some suggestions on which you can work to improve!!",
               bg = "white",font=('Times', 20))
  
    label3.pack(pady = 5)



    label2 = Label( root, text = f"{ans}",
               bg = "white",font=('Times', 20),wraplength=1500)
  
    label2.pack(pady = 50)

   

   

   
    # Show the accuracy of the classifier in a message box
    #messagebox.showinfo("Suggestions", f"{ans}")

    
# Create the main window
root = Tk()

width = root.winfo_screenwidth()
height = root.winfo_screenheight()
root.geometry(f"{width}x{height}")
root.configure(bg='white')


#root.geometry("1400x1400+0+0")
root.title("Sentiment Analysis")
#train_classifier()
# Create a button to train the classifier

label1 = Label( root, text = "Get Suggesstions For Your Organization",
               bg = "white",font=('Times', 24))
  
label1.pack(pady = 50)

train_button = Button(root, text="Upload Review File",height= 2, width=20,font=('Times', 15), pady=5,command=train_classifier)
train_button.pack()


root.mainloop()
